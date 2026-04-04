from torch import nn


class FCN(nn.Module):
    def __init__(self, in_channel,input_size,classes):
        super(FCN, self).__init__()
        self.instancenorm0 = nn.InstanceNorm1d(in_channel, affine=True)
        self.conv1d_1 = nn.Sequential(nn.Conv1d(in_channel, 128, kernel_size=8, stride=1, dilation=1,padding='same'),
                                      nn.BatchNorm1d(128),
                                      nn.ReLU(),
                                      nn.Conv1d(128, 256, kernel_size=5, stride=1, dilation=1,padding='same'),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU(),
                                      nn.Conv1d(256, 128, kernel_size=3, stride=1, dilation=1,padding='same'),
                                      nn.BatchNorm1d(128),
                                      nn.ReLU(),
                                      )
        self.avg_pooling = nn.AvgPool1d(input_size)
        self.dense = nn.Linear(128,classes)

    def forward(self, x):

        # x = self.instancenorm0(x)
        out = self.conv1d_1(x)
        repre = self.avg_pooling(out).squeeze(2)
        out = self.dense(repre)
        return out