import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from policy import action_sample, get_cost, Policy, my_get_cost
from ortools_mtsp import my_solve_mtsp
import labellines
import random
import os
import pickle

def read_var(file):
    vars = []
    with open(file) as f:
        for line in f.readlines():
            words = line.split()
            if 'var' in words:
                vars.append(float(words[-1]))
            else:
                pass
    return np.array(vars)


def plot_var(file_lax, file_reinforce, ax, n_city):
    file_lax = file_lax
    file_reinforce = file_reinforce
    vars_lax = read_var(file_lax)
    vars_reinforce = read_var(file_reinforce)
    min_len = np.min([len(vars_reinforce), len(vars_lax)])
    vars_reinforce = vars_reinforce[0:min_len]
    vars_lax = vars_lax[0:min_len]
    # y_max = np.max([np.max(vars_lax), np.max(vars_reinforce)])
    # y_min = np.max([np.min(vars_lax), np.min(vars_reinforce)])
    x = np.linspace(0, min_len, num=min_len)

    # plt.figure(figsize=(6, 5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both')
    # ax.set_ylim([y_min,y_max])
    ax.plot(x, vars_lax, label='iMTSP', linewidth=2.5)
    ax.plot(x, vars_reinforce, label='RL baseline', linewidth=2.5)
    ax.set_ylabel('Variance',fontsize=15)
    ax.set_xlabel('Iterations',fontsize=15)
    # ax.xaxis.set_tick_params(labelsize=15)
    # ax.yaxis.set_tick_params(labelsize=15)
    ax.set_title('Variance history of {} cities'.format(n_city), fontsize=15)
    ax.legend(fontsize=15)

# def get_coordinate()

def plot_tours_learning(model, dataset, device):
    model.to(device)
    model.eval()

    # to batch graph
    adj = torch.ones([dataset.shape[0], dataset.shape[1], dataset.shape[1]])  # adjacent matrix fully connected
    data_list = [Data(x=dataset[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t(), as_tuple=False) for i in range(dataset.shape[0])]
    batch_graph = Batch.from_data_list(data_list=data_list).to(device)

    # get pi
    pi = model(batch_graph, n_nodes=data.shape[1], n_batch=dataset.shape[0])
    # sample action and calculate log probabilities
    action, log_prob = action_sample(pi)
    # get reward for each batch
    reward, routes_idx, routes_coords, all_length = my_get_cost(action, data, n_agent)  # reward: tensor [batch, 1]
    fig, axs = plt.subplots(1, data.shape[0], squeeze=False)
    print(all_length)
    xval = [0.8, 0.2, 0.2, 0.5, 0.8]
    yoffset = [-0.2, 0.1, -0.2, 0.3, 0.2]

    for i in range(len(routes_coords)):
        for j in range(len(routes_coords[i])):
            axs[0, i].plot(routes_coords[i][j][:, 0], routes_coords[i][j][:, 1],marker='o',markersize=2.5, label='%.3f'%all_length[i][j])
        axs[0, i].set_title('RL baseline', fontsize=15)
        labellines.labelLines(axs[0, i].get_lines(), fontsize=15, color='k',fontweight='bold', align=False, xvals=xval,yoffsets=yoffset)#, xvals=xval,yoffsets=yoffset
    plt.savefig("D:/Projects/DRL-MTSP-main/RL1.pdf", bbox_inches='tight')

def plot_tours_ortools(data, n_agent, time_limit, new):
    if new:
        dist_matrix = torch.cdist(data, data, p=2)
        reward, route_idx, all_length = my_solve_mtsp(dist_matrix, n_agent, time_limit)
        pickle.dump(route_idx, open('route.p','wb'))
        pickle.dump(all_length,open('all_length.p','wb'))
    else:
        route_idx = pickle.load(open('route.p','rb'))
        all_length = pickle.load(open('all_length.p','rb'))
    print(all_length)
    route_coords = []
    for i in range(len(route_idx)):
        route_coords.append(data[route_idx[i], :])
    fig, ax = plt.subplots(1, 1, squeeze=False)
    xval = [0.9, 0.18, 0.2, 0.9, 0.6]
    yoffset = [0.1, 0, 0, 0, 0]
    for i in range(len(route_coords)):
        ax[0,0].plot(route_coords[i][:, 0], route_coords[i][:, 1],marker='o',markersize=2.5, label='%.3f'%all_length[i])
        ax[0,0].set_title('ORTools', fontsize=15)
    labellines.labelLines(ax[0, 0].get_lines(), fontsize=15, color='k', fontweight='bold', align=False, xvals=xval)
    plt.savefig("D:/Projects/DRL-MTSP-main/ORTools1.pdf", bbox_inches='tight')


if __name__ == '__main__':
    manual_seed = 1
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmarks = False
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    # fig.set_figheight(5)
    # plot_var('./log_var_lax1.txt', './log_var_reinforce1.txt', axs[0], 100)
    # plot_var('./log_var_lax.txt', './log_var_reinforce.txt', axs[1], 50)
    # plt.savefig("var_hist.pdf", bbox_inches='tight')
    # Code under here plots tours
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_agent = 5
    n_node_train = 100
    n_nodes = 500
    batch_size = 2
    seed = 1
    time_limit = 1800
    names = ['iMTSP', 'RL', 'ORTools', 'Var']
    name = 'iMTSP'
    if name == 'iMTSP':
        policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=32,
            key_size_policy=128, val_size=16, clipping=10, dev=dev)
        path = './saved_model/iMTSP_{}.pth'.format(str(n_node_train) + '_' + str(n_agent))
        policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    elif name == 'RL':
        policy = Policy(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                        key_size_policy=64, val_size=64, clipping=10, dev=dev)
        path = './saved_model/RL_{}.pth'.format(str(n_node_train) + '_' + str(n_agent))
        policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    elif name in ['ORTools', 'Var']:
        pass
    else:
        raise KeyError('name not defined')
    
    if name in ['iMTSP', 'RL', 'ORTools']:
        testing_data = torch.load('./testing_data/testing_data_' + str(n_nodes) + '_' + str(batch_size))
        for j in range(1):
            data = testing_data[j]
            if name in ['iMTSP', 'RL']:
                plot_tours_learning(policy, data.unsqueeze(0), dev)
            else:
                plot_tours_ortools(data, n_agent, time_limit, False)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
        fig.set_figheight(5)
        plot_var('./log_var_lax1.txt', './log_var_reinforce1.txt', axs[0], 100)
        plot_var('./log_var_lax.txt', './log_var_reinforce.txt', axs[1], 50)
        plt.savefig("var_hist.pdf", bbox_inches='tight')


