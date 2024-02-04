import torch
import torch.nn.functional as F


def assemble_feature(log_prob, action, data, no_agent, dev):
    action_onehot = F.one_hot(action, num_classes=no_agent).to(dev)
    features = torch.cat((action_onehot, data[:, 1:, :], log_prob[:, :, None]), dim=2).to(dev)
    return features