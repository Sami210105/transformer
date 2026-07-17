import torch
import torch.nn as nn

torch.manual_seed(42)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff): #d_ff is hidden layer size
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x): # x is of shape (batch, seq_len, d_model)        
        return self.net(x)
    
class AddNorm(nn.Module): # residual connection + layer normalization
    
    def __init__(self,d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) #normalizes across last dimension, so each token is normalized independently
        
    def forward(self, x, sublayer_output): # x is og input, sublayer_output is output of feedforward or self attention
        return self.norm(x + sublayer_output)
    
if __name__ == "__main__":
    batch = 1
    seq_len = 4
    d_model = 8
    d_ff = 32 # 4*d_model, hidden layer size of feedforward network
    
    x = torch.randn(batch, seq_len, d_model)
    print(f"Input x shape: {x.shape}")
    print(f"Input x: {x}")
    
    ffn = FeedForward(d_model, d_ff)
    ffn_output = ffn(x)
    print(f"FeedForward output shape: {ffn_output.shape}")
    print(f"FeedForward output: {ffn_output}")
    
    addnorm = AddNorm(d_model)
    final_output = addnorm(x, ffn_output)
    print(f"Final output after AddNorm shape: {final_output.shape}")
    print(f"Final output after AddNorm: {final_output}")
    
    