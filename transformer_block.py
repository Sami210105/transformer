from attention import MultiHeadAttention
from feedforward import FeedForward
from feedforward import AddNorm

class TransformerBlock(nn.Module):

    def __init__(self,d_model,n_heads,d_ff):

        self.attn = MultiHeadAttention(d_model,n_heads)

        self.ffn = FeedForward(d_model,d_ff)

        self.addnorm1 = AddNorm(d_model)

        self.addnorm2 = AddNorm(d_model)