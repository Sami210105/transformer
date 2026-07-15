#Final embedding = word embedding + position encoding, so the word knows where it comes in the sequence
import torch
import torch.nn as nn
import math

torch.manual_seed(42)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model) #create a matrix of zeros with length of sentence and embedding size
        position = torch.arange(0, max_len).unsqueeze(1).float() #create a vector of positions from 0 to max_len 
        #unsqueeze changes shape from (100,) to (100,1) that is column vector
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe) #register_buffer is used to register a tensor as a buffer
        
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        pos_encoding = self.pe[:, :seq_len, :]
        output = x + pos_encoding
        print(f"Output: {output.shape}")
        return output
    
if __name__ == "__main__":
    batch = 1
    seq_len = 4
    d_model = 8
    
    x = torch.zeros(batch, seq_len, d_model) 
    pos_enc = PositionalEncoding(d_model, max_len=100)
    output = pos_enc(x)
    print(output[0])
        
        