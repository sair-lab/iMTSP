from policy import Policy, action_sample, get_cost
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
import wandb
from ortools_mtsp import my_solve_mtsp

def deep_test(n_agent, n_nodes, name, device):
    for size in n_nodes:
        data     = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))
        adj      = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        rewards  = []
        # Set up model
        if name == 'RL':
            model    = Policy(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                            key_size_policy=64, val_size=64, clipping=10, dev=dev)
            path     = './saved_model/RL_{}.pth'.format(str(size) + '_' + str(n_agent))
        elif name == 'iMTSP':
            model = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=32,
                key_size_policy=128, val_size=16, clipping=10, dev=dev)
            path = './saved_model/iMTSP_{}.pth'.format(str(size)+ '_' +str(n_agent))
        else:
            KeyError('name is not correct')
        model.load_state_dict(torch.load(path, map_location=torch.device(dev)))
        model.to(device)
        model.eval()
        for i in range(batch_size):
            # to batch graph
            data_list        = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t(), as_tuple=False) for i in range(data.shape[0])]
            batch_graph      = Batch.from_data_list(data_list=data_list).to(device)
            # get pi
            pi               = model(batch_graph, n_nodes=data.shape[1], n_batch=data.shape[0])
            # sample action and calculate log probabilities
            action, log_prob = action_sample(pi)
            # get reward for each batch
            reward           = get_cost(action, data[i], n_agent)  # reward: tensor [batch, 1]
            rewards.append(reward)
            print('Max sub-tour length for instance', i, 'is', reward, 'Mean obj so far:', format(np.array(rewards).mean(), '.4f'))
        print('Size: {}, mean max length: {}'.format(size, np.array(rewards).mean()))

def ORTools_test(n_agent, n_nodes, time_limits, batch_size):
    for size in n_nodes:
        print(f'Test size: {size}')
        testing_data = testing_data = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))  
        for time_limit in time_limits:
            print(f'Time limit:{time_limit}')
            costs = []
            for i in range(batch_size):
                data                     = testing_data[i]
                dist_matrix              = torch.cdist(data, data, p=2)
                max_route_distance, _, _ = my_solve_mtsp(dist_matrix, n_agent, time_limit=time_limit)
                costs.append(max_route_distance)
                print(f'Testing instance {i}')
            print(f'Mean max length with {time_limit} seconds budget is {np.array(costs).mean()}')


if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_agent     = 5
    n_nodes     = [400, 500, 600, 700, 800, 900, 1000]
    batch_size  = 3
    time_limits = [60]
    seed        = 1
    torch.manual_seed(seed)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project =   "mtsp1",

        # track hyperparameters and run metadata
        config  =   {
            'optim': 'ORTools',
            'batch size': batch_size,
            'seed': seed
        }
    )
    names = ['iMTSP', 'RL', 'ORTools']
    name  = 'iMTSP'
 
    if name in ['iMTSP', 'RL']:
        deep_test(n_agent, n_nodes, name, dev)
    elif name == 'ORTools':
        ORTools_test(n_agent, n_nodes, time_limits, batch_size)
    else:
        raise KeyError('name not defined')



