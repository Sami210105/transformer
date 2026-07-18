import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from ffn_addnorm import FeedForward, AddNorm

class TransformerBlock(nn.Module):

    def __init__(self,d_model,n_heads,d_ff):
        super().__init__()

        self.attn = MultiHeadAttention(d_model,n_heads)

        self.ffn = FeedForward(d_model,d_ff)

        self.addnorm1 = AddNorm(d_model)

        self.addnorm2 = AddNorm(d_model)
        
    def forward(self, x):
        attn_output = self.attn(x)
        
        x = self.addnorm1(x, attn_output)
        
        ffn_output = self.ffn(x)
        
        x = self.addnorm2(x, ffn_output)
        
        return x
    
if __name__ == "__main__":
    batch = 1
    seq_len = 4
    d_model = 8
    n_heads = 2
    d_ff = 32 
    
    x = torch.randn(batch, seq_len, d_model)
    transformer_block = TransformerBlock(d_model, n_heads, d_ff)
    output = transformer_block(x)
    
    print(f"Transformer block output shape: {output.shape}")
    print(f"Transformer block output: {output}")
    
    assert output.shape == x.shape
    print("Input and output shapes match!")