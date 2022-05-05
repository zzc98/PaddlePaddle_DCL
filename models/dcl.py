import os
import paddle
import paddle.nn as nn
from paddle.vision.models import resnet


class DCLNet(nn.Layer):
    def __init__(self, config):
        super(DCLNet, self).__init__()
        self.num_classes = config.numcls
        # 主干网络
        self.model = resnet.resnet50()
        if os.path.exists(config.pretrained_model):
            self.model.load_dict(paddle.load(config.pretrained_model))
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))

        # 图像分类层、判断图像是否“破坏”、图像patch对齐
        new_layer_weight = paddle.ParamAttr(learning_rate=10)
        self.classifier = nn.Linear(2048, self.num_classes, bias_attr=False, weight_attr=new_layer_weight)
        self.classifier_swap = nn.Linear(2048, 2 * self.num_classes, bias_attr=False, weight_attr=new_layer_weight)
        self.conv_mask = nn.Conv2D(2048, 1, 1, 1, 0, bias_attr=False, weight_attr=new_layer_weight)
        self.avg_pool2 = nn.AvgPool2D((2, 2), stride=2, exclusive=False)

    def forward(self, x):
        x = self.model(x)
        mask = self.conv_mask(x)
        mask = self.avg_pool2(mask)
        mask = paddle.tanh(mask)
        mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2] * mask.shape[3]))
        x = self.avg_pool(x)
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        cls_out = self.classifier(x)
        cls_swap = self.classifier_swap(x)
        out = [cls_out, cls_swap, mask]
        return out

