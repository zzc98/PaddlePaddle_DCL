import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Layer):
    def __init__(self, reduction='mean', epsilon=0.1, ignore_index=-1):
        super().__init__()
        self.class_aixs = 1
        self.epsilon = epsilon
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(axis=self.class_aixs)
        self.nll_loss = nn.NLLLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, preds, target):
        n_class = preds.shape[self.class_aixs]
        log_preds = self.log_softmax(preds)
        target_loss = self.nll_loss(log_preds, target)
        target_loss = target_loss * (1 - self.epsilon - self.epsilon / (n_class - 1))
        untarget_loss = -log_preds.sum(axis=self.class_aixs) * (self.epsilon / (n_class - 1))
        loss = target_loss + untarget_loss
        loss = loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss
        return loss
