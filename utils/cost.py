import torch

def identity_loss(X_yng, pred_old, age_gap, age_range = 100):
    epsilon = torch.exp(-age_gap.sum(axis = 1) / age_range)
    
    # epsilon =1
    l1_loss = epsilon * torch.abs(pred_old - X_yng).mean(axis=(1, 2, 3))

    return l1_loss.mean()

def self_rec_loss(X_yng, pred_old):
    return torch.abs(X_yng - pred_old).mean()