import torch
import torch.nn as nn

class EmotionModel(nn.Module):
    def __init__(self, input_size):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 8)  # Ensure the correct number of output classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
