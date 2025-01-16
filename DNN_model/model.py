import torch
import torch.nn as nn

class DenseModel(nn.Module):
    def __init__(self, input_dim, num_classes, activation=nn.ReLU(), dropout_rate=0.3):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout(self.activation(self.bn2(self.fc2(x))))
        x = self.dropout(self.activation(self.bn3(self.fc3(x))))
        x = self.fc4(x)  # No softmax, as it's handled by the loss function
        return x

def create_model(input_dim, num_classes, lr=0.001):
    model = DenseModel(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer
