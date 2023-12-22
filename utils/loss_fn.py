import torch.nn as nn

class LossFN():
    def __init__(self) -> None:
        self.name = "loss_fn"

    def bceloss(self, inputs, targets):
        bce = nn.BCELoss()
        loss = bce(inputs, targets)
        return loss
    
    def celoss(self, inputs, targets):
        ce = nn.CrossEntropyLoss()
        loss = ce(inputs, targets)
        return loss