import numpy as np
import torch
from torch import nn

from models.SHINE.decoders import noise_decoder, trend_decoder, seasonal_decoder, dynamic_seasonal_decoder
from models.SHINE.encoders import Inception1d, Dilated1d, ts_encoder, noise_encoder
from utils import reparametrize


class SHINE(nn.Module):

    def __init__(self, ts_length, z_dim, f_dim, num_class, device='cuda', stride=1, back_bone='dilated', pooled=True,
                 K=256,P=4) -> None:
        super(SHINE, self).__init__()
        self.z_dim = z_dim
        self.ts_length = ts_length
        self.encode_noise = self.perturb = self.split_st = self.sonly = self.tonly = False

        self.pooled = pooled
        # self.smooth = MovingAverage(f_dim,10)
        if back_bone == 'inception':
            self.backbone = Inception1d(f_dim, 41, 6, use_residual=True, bottleneck_size=32)
        else:
            self.backbone = Dilated1d(f_dim, stride, self.pooled)

        t = np.arange(0, ts_length) / ts_length
        trend_degree = P
        self.T = torch.tensor(np.array([t ** i for i in range(trend_degree)])).float().to(device)

        p1, p2 = (K // 2, K // 2) if K % 2 == 0 else (K // 2, K // 2 + 1)
        s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
        s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
        self.S = torch.cat([s1, s2]).to(device)
        # self.dynamic_seasonal_decoder = dynamic_seasonal_decoder(128, z_dim, f_dim, ts_length)

        self.encode_noise = True
        self.noise_encoder = noise_encoder(f_dim, ts_length, ts_length, self.pooled)
        self.noise_decoder = noise_decoder(f_dim, ts_length, ts_length)

        self.trend_encoder = ts_encoder(ts_length, z_dim, f_dim, self.pooled)
        self.seasonal_encoder = ts_encoder(ts_length, z_dim, f_dim, self.pooled)
        self.trend_decoder = trend_decoder(trend_degree, z_dim, f_dim)  #
        self.seasonal_decoder = seasonal_decoder(K, z_dim, f_dim)
        self.constrain_S = True if K != 0 else False
        self.constrain_T = True if P !=0 else False
        if self.constrain_S is not True:
            self.seasonal_decoder = seasonal_decoder(ts_length, z_dim, f_dim)
        if self.constrain_T is not True:
            self.trend_decoder = trend_decoder(ts_length, z_dim, f_dim)

        self.classification = nn.Linear(2 * z_dim, num_class)
        self.split_st = True

    def sw(self, weak_input, strong_input):
        weak_hidden = self.backbone(weak_input)
        if strong_input is not None:
            strong_hidden = self.backbone(strong_input)
        else:
            strong_hidden = weak_hidden
        if self.training:
            if self.split_st:
                '''trend seasonal encoded from weak shifted'''
                trend_z = self.trend_encoder(weak_hidden)
                seasonal_z = self.seasonal_encoder(weak_hidden)
                x_learned_trend_coef = self.trend_decoder(trend_z)
                x_learned_trend = x_learned_trend_coef.matmul(self.T) if self.constrain_T else x_learned_trend_coef

                x_learned_seasonal_coef = self.seasonal_decoder(seasonal_z)
                x_learned_seasonal = x_learned_seasonal_coef.matmul(self.S) if self.constrain_S else x_learned_seasonal_coef

                x_learned = x_learned_trend + x_learned_seasonal
                pred_out = self.classification(torch.concat((trend_z, seasonal_z), dim=1))

            if self.encode_noise:
                '''noise is encoded from strong shifted data'''
                mu_noise, log_var_noise = self.noise_encoder(strong_hidden)
                z_noise = reparametrize(mu_noise, log_var_noise)
                x_noise_learned = self.noise_decoder(z_noise)
                x_recon = x_learned + x_noise_learned
            else:
                x_recon = x_learned
                mu_noise = log_var_noise = x_noise_learned = None
            return x_recon, x_noise_learned, x_learned_trend, x_learned_seasonal, mu_noise, log_var_noise, pred_out
        else:  # inference
            if self.split_st:
                '''trend seasonal encoded from weak shifted'''
                trend_z = self.trend_encoder(weak_hidden)
                seasonal_z = self.seasonal_encoder(weak_hidden)
                # x_learned_trend_coef = self.trend_decoder(trend_z)
                # x_learned_trend = x_learned_trend_coef.matmul(self.T) if self.constrain_T else x_learned_trend_coef

                # x_learned_seasonal_coef = self.seasonal_decoder(seasonal_z)
                # x_learned_seasonal = x_learned_seasonal_coef.matmul(self.S) if self.constrain_S else x_learned_seasonal_coef

                # x_learned = x_learned_trend + x_learned_seasonal
                print('ddddd')
                x_learned = x_learned_trend = x_learned_seasonal = None
                pred_out = self.classification(torch.concat((trend_z, seasonal_z), dim=1))
            else:
                z = self.encoder(weak_hidden)
                x_learned = x_learned_trend = x_learned_seasonal = None
                pred_out = self.classification(z)
            return x_learned, None, x_learned_trend, x_learned_seasonal, None, None, pred_out, torch.concat(
                (trend_z, seasonal_z), dim=1)

    # input: B F T
    def forward(self, weak_input, strong_input):
        return self.sw(weak_input, strong_input)
