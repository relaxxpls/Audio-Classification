import torch
import torch.nn as nn


class AudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        # self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h0, c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=self.device
        )
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=self.device
        )

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
