import torch
import torch.nn.functional as F


def d_loss(pred_real, pred_fake):
    loss_real = torch.relu(1 - pred_real).mean()
    loss_fake = torch.relu(1 + pred_fake).mean()
    loss = loss_real + loss_fake
    return loss


def g_loss(pred):
    return -pred.mean()


def feature_map_loss(fm_real, fm_fake):
    loss = 0
    for m_r, m_f in zip(fm_real, fm_fake):
        loss += F.l1_loss(m_r.detach(), m_f)
    return loss
