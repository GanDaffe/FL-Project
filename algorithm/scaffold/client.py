from algorithm.base.client import BaseClient 
from utils.train_helper import set_parameters
from import_lib import * 
from algorithm.scaffold.scaffold_utils import set_c_local, load_c_local
from logging import INFO, log

class SCAFFOLD_CLIENT(BaseClient):
    def __init__(self, *args, c_local, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_local = c_local

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        lr = config['learning_rate']
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

        results = self.train_scaffold(
            net=self.net,
            trainloader=self.trainloader,
            epochs=1,
            learning_rate=lr,
            device=config['device'],
            config=config,
            c_local=self.c_local,
            parameters=parameters
        )
        return self.get_parameters(self.net), len(self.trainloader.dataset), results
    
    def train_scaffold(self, net, trainloader, epochs, learning_rate, device, config, c_local, parameters):
        c_global_bytes = config['c_global']
        c_global = np.frombuffer(c_global_bytes, dtype=np.float64)
        global_weight = [param.detach().clone() for param in self.net.parameters()]
        if c_local is None:
            log(INFO, f"No cache found for c_local")
            c_local = [torch.zeros_like(param) for param in self.net.parameters()]

        net.to(device)  # Move model to GPU if available TODO: Make everything work on GPU.
        net.train()
        
        loss_avg, running_corrects, tot_sample = 0, 0, 0

        for _ in range(epochs):
            prebatch_params = [param.detach().clone() for param in self.net.parameters()]

            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = net(images)
                loss = self.criterion(outputs, labels)
                
                predicted = torch.argmax(outputs, dim=1)
                running_corrects += torch.sum(predicted == labels).item()
                loss_avg += loss.item() 
                tot_sample += images.shape[0]

                loss.backward()
                self.optimizer.step()

            # Local updates to the client model cf. Scaffold equation (n°3)
            # Adds Scaffold computation of c_diff in parameters
            for param, y_i, c_l, c_g in zip(self.net.parameters(), prebatch_params, c_local, c_global):
                if param.requires_grad:
                    param.grad.data = y_i - (learning_rate * (param.grad.data - c_l + c_g))

        # Update local control variate
        # Declare Scaffold variables
        y_delta = []
        c_plus = []
        c_delta = []

        # Local updates to the client control variate cf. Scaffold equation (n°4)
        # Compute c_plus : Option 2
        coef = 1 / (epochs * learning_rate)
        for c_l, c_g, param_l, param_g in zip(c_local, c_global, self.net.parameters(), global_weight):
            c_plus.append(c_l - c_g + ((param_g - param_l)*coef))


        # Compute y_delta (difference of model before and after training)
        for param_l, param_g in zip(self.net.parameters(), global_weight):
            y_delta.append(param_l - param_g)

        # Erase net params with y_delta params for weight averaging in FedAvg
        for param, new_w in zip(self.net.parameters(), y_delta):
            param.data = new_w.clone().detach() 

        # Compute c_delta
        for c_p, c_l in zip(c_plus, c_local):
            c_delta.append(c_p - c_l)

        set_c_local(self.cid, c_plus)

        # Create a bytes stream for c_delta
        # Flatten list to be compatible with numpy
        c_delta_list = []
        for param in c_delta:
            c_delta_list += param.flatten().tolist()
            
        c_delta_numpy = np.array(c_delta_list, dtype=np.float64)
        # Serialize to bytes
        c_delta_bytes = c_delta_numpy.tobytes()

        results = { 
            "loss": loss_avg / tot_sample,
            "accuracy": running_corrects / tot_sample,
            "c_delta": c_delta_bytes,
        }
        return results
    
