import torch
from torch import nn


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding='same', bias=False)


def noop(x): return x
backbone_outdim = 128

class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size > 0) else noop

        self.convs = nn.ModuleList(
            [conv(bottleneck_size if (bottleneck_size > 0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss) + 1) * nb_filters), nn.ReLU())

    def forward(self, x):
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs] + [self.conv_bottle(x)], dim=1))

        # out = self.bn_relu(torch.cat([c(x) for c in self.convs], dim=1))
        return out


#
#
class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn = nn.ReLU(True)
        self.conv = conv(ni, nf, 1)
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, inp, out):
        # print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
        # input()
        return self.act_fn(out + self.bn(self.conv(inp)))


class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert ((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d == 0 else n_ks * nb_filters,
                                                  nb_filters=nb_filters, kss=kss, bottleneck_size=bottleneck_size) for d
                                 in range(depth)])

        self.sk = nn.ModuleList(
            [Shortcut1d(input_channels if d == 0 else n_ks * nb_filters, n_ks * nb_filters) for d in range(depth // 3)])

    def forward(self, x):

        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d // 3])(input_res, x)
                input_res = x.clone()
        return x


class Inception1d(nn.Module):
    '''inception time architecture'''

    def __init__(self, input_channels=8, kernel_size=41, depth=6, bottleneck_size=32, nb_filters=16,
                 use_residual=True, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu",
                 concat_pooling=True):
        super().__init__()
        assert (kernel_size >= 40)
        kernel_size = [10, 20, 40]  # was 39,19,9

        layers = [InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth,
                                    bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)]

        # head
        # head = create_head1d(n_ks * nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head,
        #                      bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        # layers.append(head)
        # layers.append(AdaptiveConcatPool1d())
        # layers.append(nn.AdaptiveAvgPool1d(1))
        # layers.append(nn.Flatten())
        # layers.append(nn.Linear(n_ks*nb_filters, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Dilated1d(nn.Module):
    def __init__(self, f_dim, stride, pooled=False) -> None:
        super(Dilated1d, self).__init__()
        '''input [B,C,L]
        L_out = floor((L_in+2 x Padding-dilation x(kernel_size-1)-1)/stride+1)
        '''
        self.dilate_conv1d_net = nn.Sequential(
            nn.Conv1d(f_dim, 128, kernel_size=3, stride=stride, dilation=1, padding='same'),  # B,64,L-2-4-7-7-10
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=stride, dilation=2, padding='same'),  # B,64,L-2-4-7-7-10-12
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, backbone_outdim, kernel_size=7, stride=stride, dilation=4, padding='same'),  # B,64,L-2-4-7-7-10-12
            nn.BatchNorm1d(backbone_outdim),
            nn.ReLU(),
            # nn.Conv1d(256, 128, kernel_size=9, stride=stride, dilation=8, padding='same'),  # B,64,L-2-4-7-7-10-12
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Conv1d(128, 64, kernel_size=11, stride=stride, dilation=16, padding='same'),  # B,64,L-2-4-7-7-10-12
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
        )
        # self.flatten = nn.Flatten()
        self.pooled = pooled
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        ''' input [B,F,L]
            hidden [B,128,L]
        '''
        hidden = self.dilate_conv1d_net(input)
        if self.pooled:
            hidden = self.pool(hidden)
        return hidden


class ts_encoder_mu_and_sigma(nn.Module):
    def __init__(self, ts_len, z_dim) -> None:
        super(ts_encoder_mu_and_sigma, self).__init__()
        self.z_dim = z_dim
        self.linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * ts_len, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, hidden):
        '''input [B,C,L]'''
        x = self.linear_net(hidden)
        return x[:, :self.z_dim], x[:, self.z_dim:]


class ts_encoder(nn.Module):
    def __init__(self, ts_len, z_dim, f_dim, pooled=False) -> None:
        super(ts_encoder, self).__init__()
        if pooled:
            ts_len = 1
        self.z_dim = z_dim
        self.linear_net = nn.Sequential(nn.Flatten(),
                                        nn.Linear(backbone_outdim * ts_len, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, z_dim),
                                        )

    def forward(self, hidden):
        x = self.linear_net(hidden)
        return x


class noise_encoder(nn.Module):
    def __init__(self, f_dim, ts_len, z_dim, pooled=False) -> None:
        super(noise_encoder, self).__init__()
        if pooled:
            ts_len = 1
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.linear_net = nn.Sequential(nn.Flatten(),
                                        nn.Linear(backbone_outdim * (ts_len), 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, z_dim * 2 * f_dim),
                                        )

    def forward(self, input):
        '''input [B,F,T]'''
        x_noise_latent = self.linear_net(input)
        return x_noise_latent[:, :self.z_dim * self.f_dim], x_noise_latent[:, self.z_dim * self.f_dim:]
