import flwr as fl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.preprocessing import get_train_data, clustering
from utils.train_helper import compute_entropy
from run import run_simulation
from torch import nn
torch.cuda.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")


if __name__ == '__main__':
    size_img = {
        'fmnist': 28, 
        'cifar10': 32, 
        'cifar100': 32,
        'emnist': 28,
        'sentimen140': None
    }
    
    input_size = {
        'fmnist': 1,
        'emnist': 1, 
        'cifar10': 3, 
        'cifar100': 3,
        'sentimen140': 1,
    }
    
    output_size = {
        'fmnist': 10, 
        'cifar10': 10, 
        'cifar100': 100, 
        'eminst': 37, 
        'sentimen140': 2, 
    }

    # ---------- HYPER PARAMETERS -------------

    run_config = {
        'dataset_name':             'fmnist',  # emnist / fmnist / cifar10 / cifar100 / sentimen140 (take long time to load)
        'model_name':               'mlp', 
        'alpha':                    [0.5, 0.7, 0.9, 1]
    }

    experiment_config = {
        'exp_name':                 f'{run_config["dataset_name"]}_{run_config["model_name"]}',
        'algo':                     'fedadpimp',  #All letters in lowercase, no space
        'num_round':                500, 
        'num_clients':              60,
        'batch_size':               32,
        'cluster_distance':         'hellinger', # hellinger / jensenshannon / cosine ... for fedadpimp experiment only
        'learning_rate':            0.1,
        'device':                   DEVICE
    }

    model_config = {
        'model_name':               run_config['model_name'], 
        'out_shape':                output_size[run_config['dataset_name']], 
        'in_shape':                 input_size[run_config['dataset_name']],
        'hidden':                   32,
        'im_size':                  size_img[run_config['dataset_name']]
    }

    # ----------- LOADING THE DATA -------------


    ids, dist, trainloaders, testloader, client_dataset_ratio = get_train_data(dataset_name=run_config['dataset_name'], 
                                                                               num_clients=experiment_config['num_clients'], 
                                                                               batch_size=experiment_config['batch_size'],
                                                                               alphas=run_config['alpha']
                                                                            )
    client_cluster_index, distrib_ = clustering(dist, 
                                                distance=experiment_config['cluster_distance'], 
                                                min_smp=2,
                                                xi=0.15)

    num_cluster = len(list(set(client_cluster_index.values()))) - 1
    print(f'Number of Clusters: {num_cluster}')
    
    inc = 1
    for k, v in client_cluster_index.items():
        if v == -1:
            client_cluster_index[k] = num_cluster + inc
            inc += 1
            
    for k, v in client_cluster_index.items():
        print(f'Client {k + 1}: Cluster: {v}')

    for i in range(experiment_config['num_clients']):
        print(f"Client {i+1}: {dist[i]}")

    entropies = [compute_entropy(dist[i]) for i in range(experiment_config['num_clients'])]


    # ------------ RUN SIMULATION ---------------
    
    run_simulation(
        trainloaders,
        testloader, 
        client_cluster_index, 
        nn.CrossEntropyLoss(), 
        experiment_config, 
        entropies, 
        model_config,
        client_dataset_ratio
    )