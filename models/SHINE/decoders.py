import numpy as np
import torch
from torch import nn


class ts_decoder(nn.Module):
    def __init__(self, ts_len, z_dim, f_dim, stride, out_len) -> None:
        super(ts_decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            # nn.Linear(256, 64 * (ts_len-20)),
            nn.Linear(256, 64 * (ts_len)),
            nn.Tanh(),
            # nn.Unflatten(dim=1, unflattened_size=(64,(ts_len-20))),
            nn.Unflatten(dim=1, unflattened_size=(64, (ts_len))),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=stride, padding=4),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=stride, padding=4),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=stride, padding=1),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.ConvTranspose1d(8, f_dim, kernel_size=3, stride=stride, padding=1),
            nn.Tanh(),
            nn.Linear(ts_len, out_len)
        )

        # self.net = nn.Sequential(nn.Unflatten(dim=1,unflattened_size=(f_dim,z_dim)),
        #                          nn.Linear(z_dim,64),  #B,256
        #                          nn.Tanh(),
        #                          nn.Linear(64,256),
        #                          nn.Tanh(),
        #
        #                          nn.Linear(256, ts_len-20),
        #                          nn.ConvTranspose1d(f_dim, f_dim, kernel_size=13),
        #                          nn.BatchNorm1d(f_dim),
        #                          nn.Tanh(),
        #                          nn.ConvTranspose1d(f_dim, f_dim, kernel_size=7),
        #                          nn.BatchNorm1d(f_dim),
        #                          nn.Tanh(),
        #                          nn.ConvTranspose1d(f_dim, f_dim, kernel_size=3),
        #                          nn.BatchNorm1d(f_dim),
        #                          nn.Tanh(),
        #                          nn.ConvTranspose1d(f_dim, f_dim, kernel_size=1),
        #                          nn.BatchNorm1d(f_dim),
        #                          nn.Tanh(),
        #                          )

    def forward(self, input):
        x = self.net(input)
        return x


class noise_decoder(nn.Module):
    def __init__(self, f_dim, ts_len, z_dim) -> None:
        super(noise_decoder, self).__init__()
        self.linear_net = nn.Sequential(nn.Unflatten(dim=1, unflattened_size=(f_dim, z_dim)),
                                        nn.Linear(z_dim, 64),  # B,F,64
                                        nn.Tanh(),
                                        nn.Linear(64, 256),  # B,256 #B,F,256
                                        nn.Tanh(),
                                        nn.Linear(256, ts_len)  # B,F,T
                                        )

    def forward(self, input):
        '''input shape [B,F*Z]'''
        x = self.linear_net(input)
        return x


class trend_decoder(nn.Module):
    def __init__(self, P, z_dim, f_dim) -> None:
        super(trend_decoder, self).__init__()
        self.linear_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.Tanh(),
            nn.Linear(64, P * f_dim),
            nn.Unflatten(dim=1, unflattened_size=(f_dim, P)),
            # nn.Tanh(),
            # # nn.Linear(256, 64 * (ts_len-20)),
            # nn.Linear(256, 4),
        )

    def forward(self, input):
        '''input shape [B,F*Z]'''
        x = self.linear_net(input)
        return x


class seasonal_decoder(nn.Module):
    def __init__(self, ts_length, z_dim, f_dim) -> None:
        super(seasonal_decoder, self).__init__()
        self.linear_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            # # nn.Linear(256, 64 * (ts_len-20)),
            nn.Linear(256, ts_length * f_dim),
            nn.Unflatten(dim=1, unflattened_size=(f_dim, ts_length)),
        )

    def forward(self, input):
        '''input shape [B,F*Z]'''
        x = self.linear_net(input)
        return x

class dynamic_seasonal_decoder(nn.Module):
    def __init__(self, coef_num, z_dim, f_dim, ts_length) -> None:
        super(dynamic_seasonal_decoder, self).__init__()
        self.linear_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            # # nn.Linear(256, 64 * (ts_len-20)),
            nn.Linear(256, coef_num * f_dim * 4),
            nn.Unflatten(dim=1, unflattened_size=(f_dim, coef_num * 4)),  # B,chanels, coef_num
        )

        self.t = torch.tensor(np.linspace(0, ts_length, ts_length)).to('cuda').to(torch.long)

    def forward(self, z_s):
        coefs = self.linear_net(z_s)
        B, channels, coef_num = coefs.shape
        coef_nums = coef_num // 4
        A = coefs[..., :coef_nums]
        F = coefs[..., coef_nums:2 * coef_nums]
        Phi = coefs[..., 2 * coef_nums:3 * coef_nums]
        shift = coefs[..., 3 * coef_nums:]

        Phi = Phi.reshape(B, channels, coef_nums, 1)
        F = F.reshape(B, channels, coef_nums, 1)

        sin_waves = A.unsqueeze(-1) * torch.sin(np.pi * 2 * F * self.t + Phi)
        # shift =
        # self.t = self.t - shift
        # morlet_waves = A.unsqueeze(-1) * torch.cos(np.pi * 2 * F * self.t) * torch.exp(-self.t ** 2 / (2 * Phi ** 2))
        seasonality = sin_waves.sum(2)
        return A, F, Phi, seasonality
