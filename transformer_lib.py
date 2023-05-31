import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """Attention Head for the Transformer model."""
    
    def __init__(self, residual_dim, head_dim, num_tokens):
        super().__init__()
        self.scale = head_dim ** 0.5
        self.query = nn.Linear(residual_dim, head_dim)
        self.key = nn.Linear(residual_dim, head_dim)
        self.value = nn.Linear(residual_dim, head_dim)
        self.out = nn.Linear(head_dim, head_dim)
        self.register_buffer("mask", torch.tril(torch.ones(num_tokens, num_tokens)))

    def forward(self, x):
        """Implements forward pass for the Attention Head.
        
        Args:
            x (torch.Tensor): input tensor.
            
        Returns:
            torch.Tensor: output tensor.
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # compute attention weights, in the einsum
        # b = batch,
        # s = source token,
        # t = target token,
        # h = head dimension
        attention_weights = torch.einsum("bsh,bth->bst", query, key) / self.scale
        attention_weights = attention_weights.masked_fill(self.mask == 0, float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1)

        out = torch.einsum("bst,bth->bsh", attention_weights, value)
        out = self.out(out)

        return out


class MLP(nn.Module):
    """Feed Forward Network for the Transformer model."""
    
    def __init__(self, residual_dim, mlp_dim):
        super().__init__()
        self.linear_in = nn.Linear(residual_dim, mlp_dim)
        self.linear_out = nn.Linear(mlp_dim, residual_dim)
        
    def forward(self, x):
        """Implements forward pass for the Feed Forward Network.
        
        Args:
            x (torch.Tensor): input tensor.
            
        Returns:
            torch.Tensor: output tensor.
        """
        x = F.relu(self.linear_in(x))
        x = self.linear_out(x)
        return x
    

class TransformerLayer(nn.Module):
    """Transformer layer with multiple attention heads and a feed forward network."""
    
    def __init__(self, num_heads, residual_dim, head_dim, num_tokens, mlp_dim):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(residual_dim, head_dim, num_tokens) for _ in range(num_heads)])
        self.feed_forward = MLP(residual_dim, mlp_dim)
        
    def forward(self, x):
        """Implements forward pass for the Transformer Layer.
        
        Args:
            x (torch.Tensor): input tensor.
            
        Returns:
            torch.Tensor: output tensor.
        """
        x_res = x + torch.cat([head(x) for head in self.attention_heads], dim=-1)
        x_out = x_res + self.feed_forward(x_res)
        return x_out


class TransformerModel(nn.Module):
    """Transformer model with embedding, multiple layers, and unembedding."""
    
    def __init__(self, residual_dim, vocab_size, num_tokens, num_layers=2, num_heads=4, mlp_dim=64):
        super().__init__()
        assert residual_dim % num_heads == 0, "residual_dim must be divisible by num_heads"

        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(vocab_size, residual_dim)
        self.pos_embedding = nn.Embedding(num_tokens, residual_dim)
        self.layers = nn.ModuleList([TransformerLayer(num_heads, residual_dim, residual_dim // num_heads, num_tokens, mlp_dim) for _ in range(num_layers)])
        self.unembedding = nn.Linear(residual_dim, vocab_size)

    def forward(self, x):
        """Implements forward pass for the Transformer Model.
        
        Args:
            x (torch.Tensor): input tensor.
            
        Returns:
            torch.Tensor: output tensor.
        """
        x = self.embedding(x) + self.pos_embedding(torch.arange(self.num_tokens, device=x.device))

        for layer in self.layers:
            x = layer(x)

        x = self.unembedding(x)
        return x
