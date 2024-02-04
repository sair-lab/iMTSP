from policy import Policy, action_sample, get_cost
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
import wandb
import time

def test(model, dataset, n_agent, device):

    # model to device
    model.to(device)
    model.eval()

    # to batch graph
    adj = torch.ones([dataset.shape[0], dataset.shape[1], dataset.shape[1]])  # adjacent matrix fully connected
    data_list = [Data(x=dataset[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t(), as_tuple=False) for i in range(dataset.shape[0])]
    batch_graph = Batch.from_data_list(data_list=data_list).to(device)
    start = time.time()
    # get pi
    pi = model(batch_graph, n_nodes=data.shape[1], n_batch=dataset.shape[0])
    # sample action and calculate log probabilities
    action, log_prob = action_sample(pi)
    # get reward for each batch
    reward = get_cost(action, data, n_agent)  # reward: tensor [batch, 1]
    end = time.time()
    times.append(end-start)
    return np.array(reward).mean()


if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_agent = 15
    n_city_train = 100
    n_nodes = [1000]
    batch_size = 512
    seed = 86
    print(n_agent, n_city_train)
    print(n_nodes)
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mtsp1",
        # set resume configuration
        # id='k7e1ksoi',
        # resume='allow',  
        # track hyperparameters and run metadata
        config={
            'stage':'test',
            'optim':'LAX',
            'n_agent':n_agent,
            'n_city_train':n_city_train,
            'batch size':batch_size
        }
    )
    torch.manual_seed(seed)
    # load net
    # Check policy and surrogate setting!!!!!
    # policy = Policy(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
    #                 key_size_policy=64, val_size=64, clipping=10, dev=dev)
    policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=32,
                key_size_policy=128, val_size=16, clipping=10, dev=dev) # 32 64 64 64
    path = './saved_model/lax_newpolicy_noble_{}_{}.pth'.format(str(n_city_train), str(n_agent))
    policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    for size in n_nodes:
        testing_data = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))
        print('Size:', size)
        objs = []
        times = []
        for j in range(15):
            # data = torch.rand(size=[1, size, 2])  # [batch, nodes, fea], fea is 2D location
            data = testing_data[j].unsqueeze(0)
            # testing
            obj = test(policy, data, n_agent, dev)
            objs.append(obj)
            print('Max sub-tour length for instance', j, 'is', obj, 'Mean obj so far:', format(np.array(objs).mean(), '.4f'))
        print(np.mean(times))





