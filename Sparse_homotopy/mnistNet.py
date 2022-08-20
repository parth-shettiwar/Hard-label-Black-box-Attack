import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = x.reshape(-1, 28*28)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out