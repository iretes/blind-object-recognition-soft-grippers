import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)

class CNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, embedding_dim=128, dropout=0.0):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels, dropout=dropout)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels*2, embedding_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, e1, e2, sim):
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)
        dist = F.pairwise_distance(e1, e2)
        loss = sim * dist.pow(2) + (1 - sim) * F.relu(self.tau - dist).pow(2)
        return loss.mean()