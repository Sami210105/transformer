import torch
import torch.nn as nn

torch.manual_seed(42)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )