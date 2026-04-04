import torch
from torch import nn
from transformers import AutoModelForCausalLM


class ClassificationHead(nn.Module):
    def __init__(
            self,
            n_channels: int = 1,
            d_model: int = 768,
            n_classes: int = 2,
            n_patches: int = 96,
            head_dropout: int = 0.1,
            reduction: str = "mean",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.reduction = reduction
        if self.reduction == "mean" or isinstance(self.reduction, list):
            self.linear = nn.Linear(d_model, n_classes)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Linear(512, n_classes)
            )
        # Concatenate across channels
        elif self.reduction == "concat_channels":
            self.linear = nn.Linear(n_channels * d_model, n_classes)

            self.mlp = nn.Sequential(
                nn.Linear(n_channels * d_model, 512),
                nn.ReLU(),
                nn.Linear(512, n_classes)
            )

        elif self.reduction == "concat_patches":
            self.linear = nn.Linear(n_patches * d_model, n_classes)
            self.mlp = nn.Sequential(
                nn.Linear(n_patches * d_model, 512),
                nn.ReLU(),
                nn.Linear(512, n_classes)
            )
        elif self.reduction == 'weighted':
            self.logits = nn.Parameter(torch.zeros(n_channels, n_patches))

            # 分类器的最后一层，用于将加权后的特征转换为类别预测
            self.linear = nn.Linear(n_patches * d_model, n_classes)  # 根据加权后的通道特征进行分类
        else:
            raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")

    def forward(self, x):

        batch_size, n_channels, n_patches, d_model = x.shape
        if self.reduction == "mean":
            # Mean across channels
            repre = x.mean(1)  # [batch_size x d_models x n_patches]
            # Mean across patches
            repre = repre.mean(1)  # [batch_size x d_model]
        # Concatenate across channels
        elif self.reduction == "concat_channels":
            # [batch_size x n_patches x d_model * n_channels]
            repre = x.permute(0, 2, 3, 1).reshape(
                batch_size, n_patches, d_model * n_channels)
            repre = repre.mean(1)
        elif self.reduction == "concat_patches":
            # [batch_size x n_patches x d_model * n_channels]
            repre = x.reshape(
                batch_size, n_channels, d_model * n_patches)
            repre = repre.mean(1)
        elif self.reduction == 'weighted':
            weighted_features = torch.einsum("bcpd, cpdq -> bcq", x, self.weights)
            output = weighted_features.mean(dim=1)  # [batch_size, n_classes]

            # 进行 dropout
            output = self.dropout(output)

            # 输出最终的分类结果
            y = self.linear(output)
            # y = repre.mean(1)
            return y

            # 分类器的最后一层，用于将加权后的特征转换为类别预测
        elif isinstance(self.reduction, list):
            repre = x[:, self.reduction[0], self.reduction[1], :]
        else:
            raise NotImplementedError(f"Reduction method {self.reduction} not implemented.")

        x = self.dropout(repre)
        y = self.linear(x)
        # y = self.mlp(x)
        return y

class TimerWithHead(nn.Module):
    def __init__(self, num_class,n_channels,ts_len):
        super().__init__()
        timer = AutoModelForCausalLM.from_pretrained(
            'thuml/timer-base-84m',
            # device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
            trust_remote_code=True,
        )
        timer.to('cuda')
        timer.train()
        self.backbone = timer.model

        # self.backbone.apply(lambda m: hasattr(m, 'reset_parameters') and m.reset_parameters())
        self.backbone.train()

        self.patch_size =96
        self.n_patches = ts_len//self.patch_size
        self.d_model = 1024
        self.n_channels = n_channels

        self.classifier = ClassificationHead(
            n_channels=self.n_channels,
            d_model=self.d_model,
            n_classes=num_class,
            n_patches=self.n_patches,
        )

    def forward(self, x):
        B,F,T = x.shape
        x=x.reshape(B*F,T)
        output = self.backbone(x)
        embedding = output[0]
        # embedding = self.backbone.embed_layer(x)

        embedding = embedding.reshape(-1, self.n_channels, self.n_patches, self.d_model)

        logits = self.classifier(embedding)        # 线性层

        return embedding,logits