import torch
from torch.nn import functional as F


def compute_smooth_loss(x1, x2, coef):
    if x1 is None:
        return coef * F.mse_loss(x2[:, :, :-1], x2[:, :, 1:])
    elif x2 is None:
        return coef * F.mse_loss(x1[:, :, :-1], x1[:, :, 1:])
    else:
        return coef * F.mse_loss(x1[..., :-1], x1[..., 1:]) + coef * F.mse_loss(x2[..., :-1], x2[..., 1:])


def KLD(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)


def loss_function(pred_out, x_label, recons_x, input_x, x_learned_trend, x_learned_seasonal, mu_noise, log_var_noise,
                  alpha=10.0, decoder=True):
    if decoder == False:
        pred_loss = F.cross_entropy(pred_out, x_label)
        return pred_loss, None, None, None, None, None, pred_loss
    if recons_x is not None:
        recon_loss = F.mse_loss(recons_x, input_x)
    else:
        recon_loss = torch.tensor(0.0)
    pred_loss = F.cross_entropy(pred_out, x_label)

    smooth_loss = compute_smooth_loss(x_learned_trend, x_learned_seasonal, alpha)  # torch.tensor(0.0) #

    if mu_noise is not None:
        noise_loss = torch.mean(-0.5 * torch.sum(1 + log_var_noise - mu_noise ** 2 - log_var_noise.exp(), dim=1),
                                dim=0)  # N(0,1)
    else:
        noise_loss = torch.tensor(0.0)

    return recon_loss + noise_loss + smooth_loss + pred_loss, recon_loss, smooth_loss, noise_loss, pred_loss
