import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            1,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output
