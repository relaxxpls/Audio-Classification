import torch.nn as nn
import torch.nn.functional as F


class PrintLayer(nn.Module):
    def __init__(self, title=None):
        super().__init__()
        self.title = title

    def forward(self, x):
        print(self.title, x.shape)

        return x


class AudioCNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.main = nn.Sequential(
            # PrintLayer('Start'),
            nn.Conv1d(1, 64, 80, 4, 2),
            # PrintLayer('Conv1'),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 80, 4, 2),
            # PrintLayer('Conv2'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 80, 4, 2),
            # PrintLayer('Conv3'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, 80, 4, 2),
            # PrintLayer('Conv4'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, 40, 4, 2),
            # PrintLayer('Conv5'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintLayer('end'),
        )
        self.classifier = nn.Sequential(
            # PrintLayer('classifier'),
            nn.Linear(512 * 29, num_classes),
            # PrintLayer('linear'),
            nn.Softmax(dim=-1),
            # PrintLayer('softmax'),
        )

    def forward(self, tensor):
        batch_size = tensor.size(0)
        hidden = self.main(tensor)
        hidden = hidden.view(batch_size, -1)
        hidden = self.classifier(hidden)

        return hidden


class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_filters = 32
        self.num_classes = num_classes

        self.main = nn.Sequential(
            nn.Conv2d(1, self.num_filters, 11, padding=5),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_filters, self.num_filters, 3, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(self.num_filters, self.num_filters * 2, 3, padding=1),
            nn.BatchNorm2d(self.num_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_filters * 2, self.num_filters * 4, 3, padding=1),
            nn.BatchNorm2d(self.num_filters * 4),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(self.num_filters * 4, self.num_classes)

    def forward(self, x):
        x = self.main(x)

        # x = x.view(x.size(0), -1)
        x = x[:, :, 0, 0]
        x = self.fc(x)

        return x
