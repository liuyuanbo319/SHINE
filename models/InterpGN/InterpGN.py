import torch
import torch.nn as nn

from models.Conv1DBased.FCN import FCN
from models.InterpGN.Shapelet import ShapeBottleneckModel, ModelInfo


class InterpGN(nn.Module):
    def __init__(
            self,
            configs,
            num_shapelet=[5, 5, 5, 5],
            shapelet_len=[0.1, 0.2, 0.3, 0.5],
            # num_shapelet=[1],
            # shapelet_len=[0.5],
    ):
        super().__init__()

        self.configs = configs
        self.sbm = ShapeBottleneckModel(
            configs=configs,
            num_shapelet=num_shapelet,
            shapelet_len=shapelet_len
        )
        self.deep_model = FCN(configs.in_channels,configs.ts_len,configs.num_classes).to(configs.device)

    def forward(self, x, gating_value=None):
        sbm_out, model_info = self.sbm(x)
        deep_out = self.deep_model(x)

        # Gini Index: compute the gating value
        p = nn.functional.softmax(sbm_out, dim=-1)
        c = sbm_out.shape[-1]
        gini = p.pow(2).sum(-1, keepdim=True)
        sbm_util = (c * gini - 1) / (c - 1)
        if gating_value is not None:
            mask = (sbm_util > gating_value).float()
            sbm_util = torch.ones_like(sbm_util) * mask + sbm_util * (1 - mask)
        deep_util = torch.ones_like(sbm_util) - sbm_util
        output = sbm_util * sbm_out + deep_util * deep_out

        return output, ModelInfo(d=model_info.d,
                                 p=model_info.p,
                                 eta=sbm_util,
                                 shapelet_preds=sbm_out,
                                 dnn_preds=deep_out,
                                 preds=output,
                                 loss=self.loss().unsqueeze(0))

    def loss(self):
        return self.sbm.loss()

    def step(self):
        self.sbm.step()
