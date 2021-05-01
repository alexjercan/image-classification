# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, num_classes=30):
        super(Model, self).__init__()
        
        self.model = resnet50(num_classes=num_classes)

    def forward(self, img):
        return self.model(img)


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

        self.class_loss_val = 0

    def forward(self, predictions, targets):
                
        class_loss = self.loss_fn(predictions, targets)
        self.class_loss_val = class_loss.item()

        return class_loss
    
    def show(self):
        loss = self.class_loss_val
        return f'(class_loss:{loss:.4f})'


if __name__ == "__main__":
    img = torch.rand((4, 3, 256, 256))
    model = Model(num_classes=30)
    pred = model(img)
    assert pred.shape == (4, 30), f"dataset error {pred.shape}"

    print("model ok")
