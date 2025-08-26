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

    def classify(self, x):
        return torch.softmax(self.forward(x), dim=1)
    
    def log_prob_fn_extinct(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.forward(z)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs[:, 1].sum()
    
    def eval_probs(self, z) -> None:
        probs = self.classify(z)
        print(f"Diffusion Probs: {probs}")
        argmax = torch.argmax(probs, dim=1)
        print(f"Percent Extinct: {argmax.sum() / len(argmax)}")