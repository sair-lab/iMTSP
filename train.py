from policy import Policy, action_sample, get_cost
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation import validate
import numpy as np
import wandb
import os


def train(path, batch_size, no_nodes, policy_net, l_r, no_agent, iterations, device):
    # prepare validation data
    validation_data = torch.load('./validation_data/validation_data_'+str(no_nodes)+'_'+str(batch_size))
    # a large start point
    best_so_far = np.inf # change when resuming
    validation_results = []

    # optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    for itr in range(iterations):
        # prepare training data
        data = torch.load('./training_data/training_data_'+str(no_nodes)+'_'+str(batch_size)+'_'+str(itr%10))  # [batch, nodes, fea], fea is 2D location
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # get pi
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=batch_size)
        # sample action and calculate log probabilities
        action, log_prob = action_sample(pi)
        # get reward for each batch
        cost = get_cost(action, data, no_agent)  # reward: tensor [batch, 1]
        # compute loss
        loss = torch.mul(torch.tensor(cost, device=device) - 2, log_prob.sum(dim=1)).sum()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % 100 == 0:
            print('\nIteration:', itr)
        print(format(sum(cost) / batch_size, '.4f'))
        wandb.log({'cost':sum(cost) / batch_size})

        # validate and save best nets
        if (itr+1) % 100 == 0:
            validation_result = validate(validation_data, policy_net, no_agent, device)
            wandb.log({'best val so far':validation_result})
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), path)
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result
    return validation_results


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed=86
    torch.manual_seed(seed)

    n_agent = 5
    n_nodes = 50
    batch_size = 512
    lr = 1e-4
    iteration = 2500 # change when resuming

    policy = Policy(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    
    # To resume from a breakpoint with wandb. When resuming, do check hyperparameters like learning rate, best validation results
    path = './saved_model/RL_{}_{}.pth'.format(str(n_nodes), str(n_agent))
    if os.path.isfile(path):
        policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
        id = '' # This should be the mission id in wandb
    else:
        id=''
    
    wandb.login(key='') # Login with wandb account key
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mtsp",
        # # set resume configuration
        id=id,
        resume='allow',     
        # track hyperparameters and run metadata
        config={
            'stage':'train',
            'optim':'REINFORCE',
            'n_node':n_nodes,
            'n_agent':n_agent,
            "epochs": iteration,
            'seed':seed,
            'lr':lr
        }
    )
    print('run id:{}'.format(id))
    best_results = train(path, batch_size, n_nodes, policy, lr, n_agent, iteration, dev)
    print(min(best_results))
    wandb.finish()
