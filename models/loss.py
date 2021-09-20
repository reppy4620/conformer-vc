import torch


def d_loss(pred_real, pred_fake):
    loss_real = torch.relu(1 - pred_real).mean()
    loss_fake = torch.relu(1 + pred_fake).mean()
    loss = loss_real + loss_fake
    return loss


def g_loss(pred):
    return -pred.mean()
