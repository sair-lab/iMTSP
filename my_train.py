from policy import Policy, action_sample, get_cost, Surrogate
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation import validate
import numpy as np
import wandb
import os


def train(batch_size, no_nodes, policy_net, surrogate, lr_p, lr_s, no_agent, iterations, device, path, path_s):
    # prepare validation data
    validation_data = torch.load('./validation_data/validation_data_'+str(no_nodes)+'_'+str(batch_size))
    # a large start point
    best_so_far = np.inf # change when resuming
    validation_results = []

    # optimizer
    optim_p = torch.optim.RMSprop(policy_net.parameters(), lr=lr_p, momentum=0.468, weight_decay=0.067)
    optim_s = torch.optim.RMSprop(surrogate.parameters(), lr=lr_s, momentum=0.202, weight_decay=0.336)
    scheduler_p = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_p, min_lr=1e-6, patience=50, factor=0.5, verbose=True)
    scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_s, min_lr=1e-6, patience=50, factor=0.5, verbose=True)

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
        # get real cost for each batch
        cost = get_cost(action, data, no_agent)  # cost: tensor [batorch.cat([torch.reshape(p, [-1]) for p in pg_grads], 0)tch, 1]
        # estimate cost via the surrogate network
        cost_s = torch.squeeze(surrogate(log_prob))
        # compute loss, need to freeze surrogate's parameters, cost_s in the second term should be detached
        loss = torch.mul(torch.tensor(cost, device=device) - 2, log_prob.sum(dim=1)).sum() \
               - torch.mul(cost_s.detach() - 2, log_prob.sum(dim=1)).sum() \
               + (cost_s - 2).sum()

        scheduler_p.step(torch.tensor(cost, device=device).sum())
        scheduler_s.step(torch.tensor(cost, device=device).sum())
        # compute gradient's variance loss w.r.t. surrogate's parameter
        grad_p = torch.autograd.grad(loss, policy_net.parameters(),
                                     grad_outputs=torch.ones_like(loss), create_graph=True, retain_graph=True)
        grad_temp = torch.cat([torch.reshape(p, [-1]) for p in grad_p], 0)
        grad_ps = torch.square(grad_temp).mean(0)
        wandb.log({'variance': grad_ps})
        grad_s = torch.autograd.grad(grad_ps, surrogate.parameters(),
                                     grad_outputs=torch.ones_like(grad_ps), retain_graph=True, allow_unused=True)
        # Optimize the policy net
        optim_p.zero_grad()
        loss.backward()
        optim_p.step()
        # Optimize the surrogate net
        optim_s.zero_grad()
        for params, grad in zip(surrogate.parameters(), grad_s):
            params.grad = grad
        optim_s.step()
        if itr % 100 == 0:
            print('\nIteration:', itr)
        print(format(sum(cost) / batch_size, '.4f'))
        wandb.log({'cost':sum(cost) / batch_size})
        wandb.log({'diff of cost':(sum(cost) - sum(cost_s).detach()) / batch_size})

        # validate and save best nets
        if (itr+1) % 100 == 0:
            validation_result = validate(validation_data, policy_net, no_agent, device)
            wandb.log({'best val so far':validation_result})
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), path)
                torch.save(surrogate.state_dict(), path_s)
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result
    return validation_results


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.use_deterministic_algorithms(True)

    n_agent = 5 
    n_nodes = 100
    batch_size = 512
    lr_p = 1e-4 # change when resuming
    lr_s = 1e-3 # change when resuming
    iteration = 2500 # change when resuming

    seed = 86
    torch.manual_seed(seed)
    
    policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=32,
                    key_size_policy=128, val_size=16, clipping=10, dev=dev) 
    surrogate = Surrogate(in_dim=n_nodes-1, out_dim=1, n_hidden=256, nonlin='tanh', dev=dev)

    path = './saved_model/iMTSP_{}_{}.pth'.format(str(n_nodes), str(n_agent))
    path_s = './saved_surrogate/iMTSP_surrogate_{}_{}.pth'.format(str(n_nodes), str(n_agent))

    # To resume from a breakpoint with wandb. When resuming, do check hyperparameters like learning rate, best validation results
    if os.path.isfile(path):
        policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
        surrogate.load_state_dict(torch.load(path_s, map_location=torch.device(dev)))
        id = ''  # This should be the mission id in wandb
    else:
        id = ''

    # Config your wandb
    wandb.login(key='') # Login with wandb account key
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mtsp",
        # set resume configuration
        id=id,
        resume='allow',
        # track hyperparameters and run metadata
        config={
            'stage':'train',
            'optim':'LAX',
            'n_node':n_nodes,
            'n_agent':n_agent,
            "epochs": iteration,
            'lr_p':lr_p,
            'lr_s':lr_s
        }
    )
    print('run id:{}'.format(id))
    best_results = train(batch_size, n_nodes, policy, surrogate, lr_p, lr_s, n_agent, iteration, dev, path, path_s)
    print(min(best_results))
    wandb.finish()

