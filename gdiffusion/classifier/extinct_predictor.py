import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EsmClassificationHead(nn.Module):
    # slightly modified from the original ESM classification head
    def __init__(self, input_dim=256):
        super().__init__()
        self.dense = nn.Linear(input_dim, 2048)
        self.dropout = nn.Dropout(0.05)
        self.dense2 = nn.Linear(2048, 2048)
        self.dense3 = nn.Linear(2048, 2048)
        self.out_proj = nn.Linear(2048, 2)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x