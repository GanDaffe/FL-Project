import flwr as fl
from flwr.common import ndarrays_to_parameters
from utils.train_helper import get_model, get_parameters
from algorithm.scaffold.scaffold_utils import load_c_local
from import_algo import *
from algorithm.moon.moon_model import init_model

def run_simulation(
        trainloaders,
        testloader,
        client_cluster_index,
        criterion,
        exp_config,
        entropies, 
        model_config: dict,
        client_dataset_ratio
    ):
    
    algo = exp_config['algo']
    device = exp_config['device'] 
    model_name = model_config['model_name']

    net = init_model(model_name, model_config).to(device) if algo == 'moon' else get_model(model_name, model_config).to(device)

    def base_client_fn(cid: str): 
        idx = int(cid)
        return BaseClient(cid, net, trainloaders[idx], criterion).to_client()
    def fednova_client_fn(cid: str):
        idx = int(cid)
        return FedNovaClient(cid, net, trainloaders[idx], criterion, ratio=client_dataset_ratio).to_client()   
    def cluster_fed_client_fn(cid: str) -> ClusterFedClient:
        idx = int(cid)
        return ClusterFedClient(cid, net, trainloaders[idx], criterion, cluster_id=client_cluster_index[idx]).to_client()   
    def fedprox_client_fn(cid: str): 
        idx = int(cid)
        return BaseClient(cid, net, trainloaders[idx], criterion).to_client()
    def scaffold_client_fn(cid: str): 
        idx = int(cid)
        c_local = load_c_local(idx)
        return SCAFFOLD_CLIENT(cid, net, trainloaders[idx], criterion, c_local=c_local).to_client()
    def moon_client_fn(cid: str): 
        idx = int(cid)
        return MoonClient(cid, net, trainloaders[idx], criterion, dir='/moon_models').to_client()
    

    current_parameters = ndarrays_to_parameters(get_parameters(net))
    client_resources = {"num_cpus": 1, "num_gpus": 0.2} if device == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}    

    if algo == 'fednova': 
        fl.simulation.start_simulation(
            client_fn           = fednova_client_fn,
            num_clients         = exp_config['num_clients'],
            config              = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy            = FedNovaStrategy( 
                exp_name            = exp_config['exp_name'],
                learning_rate       = exp_config['learning_rate'],
                algo_name           = algo,
                net                 = net,
                testloader          = testloader,
                device              = device,
                num_rounds          = exp_config['num_round'],
                num_clients         = exp_config['num_clients'],
                current_parameters  = current_parameters,
            ),

            client_resources        = client_resources
        )
    elif algo == 'fedadpimp': 
        assert entropies != None, f'Entropies for {algo} cannnot be none'
        fl.simulation.start_simulation(
            client_fn           = cluster_fed_client_fn,
            num_clients         = exp_config['num_clients'],
            config              = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy            = BoxFedv2(
                exp_name            = exp_config['exp_name'],
                learning_rate       = exp_config['learning_rate'],
                algo_name           = algo,
                net                 = net,
                testloader          = testloader,
                device              = device,
                num_rounds          = exp_config['num_round'],
                num_clients         = exp_config['num_clients'],
                current_parameters  = current_parameters,
                entropies           = entropies,
                ),

            client_resources    = client_resources
        )   
    elif algo == 'fedprox': 
        fl.simulation.start_simulation(
            client_fn           = fedprox_client_fn,
            num_clients         = exp_config['num_clients'],
            config              = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy            = FedProx(
                exp_name            = exp_config['exp_name'],
                learning_rate       = exp_config['learning_rate'],
                algo_name           = algo,
                net                 = net,
                testloader          = testloader,
                device              = device,
                num_rounds          = exp_config['num_round'],
                num_clients         = exp_config['num_clients'],
                current_parameters  = current_parameters,
                            ),
            client_resources    = client_resources
        )
    elif algo == 'fedimp': 
        assert entropies != None, f'Entropies for {algo} cannnot be none'
        fl.simulation.start_simulation(
            client_fn           = base_client_fn,
            num_clients         = exp_config['num_clients'],
            config              = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy            = FedImp(
                exp_name            = exp_config['exp_name'],
                learning_rate       = exp_config['learning_rate'],
                algo_name           = algo,
                net                 = net,
                testloader          = testloader,
                device              = device,
                num_rounds          = exp_config['num_round'],
                num_clients         = exp_config['num_clients'],
                current_parameters  = current_parameters,
                entropies           = entropies,
                ),

            client_resources    = client_resources
        )   
    elif algo == 'fedadp':
        fl.simulation.start_simulation(
            client_fn           = base_client_fn,
            num_clients         = exp_config['num_clients'],
            config              = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy            = FedAdp(
                learning_rate       = exp_config['learning_rate'],
                exp_name            = exp_config['exp_name'],
                algo_name           = algo,
                net                 = net,
                testloader          = testloader,
                device              = device,
                num_rounds          = exp_config['num_round'],
                num_clients         = exp_config['num_clients'],
                current_parameters  = current_parameters,
                            ),

            client_resources    = client_resources
        )   
    elif algo == 'fedavg': 
        fl.simulation.start_simulation(
            client_fn           = base_client_fn, 
            num_clients         = exp_config['num_clients'],
            config              = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy            = FedAvg(
                learning_rate       = exp_config['learning_rate'],
                exp_name            = exp_config['exp_name'],
                algo_name           = algo,
                net                 = net,
                testloader          = testloader,
                device              = device,
                num_rounds          = exp_config['num_round'],
                num_clients         = exp_config['num_clients'],
                current_parameters  = current_parameters,
                ),
            client_resources     = client_resources
        )
    elif algo == 'scaffold': 
        fl.simulation.start_simulation(
            client_fn           = scaffold_client_fn,
            num_clients         = exp_config['num_clients'],
            config              = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy            = SCAFFOLD(
                learning_rate       = exp_config['learning_rate'],
                exp_name            = exp_config['exp_name'],
                algo_name           = algo,
                net                 = net,
                testloader          = testloader,
                device              = device,
                num_rounds          = exp_config['num_round'],
                num_clients         = exp_config['num_clients'],
                current_parameters  = current_parameters,
            ),

            client_resources    = client_resources
        )
    elif algo == 'moon': 
        fl.simulation.start_simulation(
            client_fn           = moon_client_fn, 
            num_clients         = exp_config['num_clients'], 
            config              = fl.server.ServerConfig(num_rounds = exp_config['num_round']), 
            strategy            = MOON(
                learning_rate       = exp_config['learning_rate'],
                exp_name            = exp_config['exp_name'],
                algo_name           = algo, 
                net                 = net,
                testloader          = testloader, 
                device              = device,
                num_rounds          = exp_config['num_round'], 
                num_clients         = exp_config['num_clients'], 
                current_parameters  = current_parameters

                ),
            client_resources = client_resources
        )
