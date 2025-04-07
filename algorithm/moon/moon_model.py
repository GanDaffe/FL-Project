from torch import nn  
import torch.nn.functional as F
from models import ResNet101, ResNet50
def calculate(size, kernel, stride, padding):
    return int(((size+(2*padding)-kernel)/stride) + 1)

class MLP_header(nn.Module): 
    
    def __init__(self) -> None:
        super(MLP_header, self).__init__()
        self.layer1 = nn.Linear(28*28, 200)
        self.layer2 = nn.Linear(200, 200)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x) 

        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        return x
        
class CNN_header(nn.Module): 
    def __init__(self, in_feat, im_size, out_feat, hidden): 
        
        super(CNN_header, self).__init__()
        out = im_size 

        self.conv1 = nn.Conv2d(in_feat, 32, kernel_size=3, stride=1, padding=1)
        out = calculate(out, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        out = calculate(out, kernel=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        out = calculate(out, 3, 1, 1)
        out = calculate(out, kernel=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=0.5)
        self.after_conv = out * out * 64
        self.fc = nn.Linear(in_features=self.after_conv, out_features=hidden) 
    
    def forward(self, X): 
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))

        X = X.view(-1, self.after_conv)
        X = F.relu(self.fc(X))
        X = self.dropout(X)

        return X
class LSTM_header(nn.Module):

    def __init__(self):
        super(LSTM_header, self).__init__()
        self.embedding = nn.Embedding(2000, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=64)

        self.fc1 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(256, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.fc2(x)
        return x
    
class ModelMoon(nn.Module): 

    def __init__(self, base_model, n_classes, model_configs=None):
        super(ModelMoon, self).__init__() 

        if base_model == 'mlp':
            self.features = MLP_header() 
            num_ftrs = 200 
        elif base_model == 'cnn': 
            self.features = CNN_header(in_feat=model_configs['in_shape'],
                                       im_size=model_configs['im_size'], 
                                       hidden=model_configs['hidden'])
            num_ftrs = model_configs['hidden'] 
        elif base_model in ['resnet50', 'resnet101']: 
            if base_model == 'resnet50': 
                model = ResNet50(num_channel=model_configs['in_shape'], num_classes=n_classes)
            else: 
                model = ResNet101(num_channel=model_configs['in_shape'], num_classes=n_classes) 
            
            self.features = nn.Sequential(*list(model.children())[:-1])
            num_ftrs = model.fc.in_features
        elif base_model == 'lstm': 
            self.features = LSTM_header()
            num_ftrs = 256 

        # self.ln1 = nn.Linear(num_ftrs, num_ftrs) 
        # self.ln2 = nn.Linear(num_ftrs, out_dim) 
        self.base_model = base_model
        self.l3 = nn.Linear(num_ftrs, n_classes) 

    def forward(self, x): 
        h = self.features(x) 
        h = h.view(h.size(0), -1)

        # x = self.ln1(h) 
        # x = F.relu(x)
        # x = self.l2(x) 
        
        y = self.l3(h) 
        if self.base_model == 'lstm': 
            y = F.sigmoid(y)
        return h, h, y

def init_model(model_name, model_config): 
    return ModelMoon(base_model=model_name,
                     n_classes=model_config['out_shape'], 
                     model_configs=model_config)
