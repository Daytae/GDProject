# Really really bad logp predictor
# predicts logp (really badly) from latent

import torch
import torch.nn as nn

class LogPPredictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(LogPPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)
