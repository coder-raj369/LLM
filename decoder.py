import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    """
    A single standard Decoder-Only Transformer block used in modern LLMs.
    Features Pre-Layer Normalization and Causal Masking.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Multi-Head Attention components
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 2. Feed-Forward Network (MLP)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # Modern standard activation function
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 3. Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _masked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes scaled dot-product attention with a causal mask."""
        B, num_heads, S, head_dim = q.shape
        
        # Calculate raw attention scores: (B, num_heads, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Create lower-triangular causal mask so tokens cannot look into the future
        mask = torch.tril(torch.ones(S, S, device=q.device)).view(1, 1, S, S)
        
        # Fill masked positions with a near-negative-infinity value
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Convert scores to probabilities and apply dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Multiply weights by context values
        context = torch.matmul(attn_weights, v)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim)
        """
        B, S, E = x.shape
        
        # --- Sublayer 1: Pre-LN & Masked Multi-Head Attention ---
        norm_x = self.norm1(x)
        
        # Project and reshape for multi-head split: (B, S, num_heads, head_dim)
        q = self.q_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention core loop: outputs shape (B, num_heads, S, head_dim)
        attn_out = self._masked_attention(q, k, v)
        
        # Permute and flatten back to original structure
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, E)
        attn_out = self.out_proj(attn_out)
        
        # Residual connection
        x = x + attn_out
        
        # --- Sublayer 2: Pre-LN & Feed-Forward Network ---
        ffn_out = self.ffn(self.norm2(x))
        
        # Final residual connection
        x = x + ffn_out
        
        return x

# Quick verification test
if __name__ == "__main__":
    # Mock inputs: batch size 2, sequence length 5, embedding size 256
    sample_input = torch.randn(2, 5, 256)
    
    # Initialize decoder block
    decoder_block = DecoderBlock(embed_dim=256, num_heads=8, ff_dim=1024)
    output = decoder_block(sample_input)
    
    print("Input Shape :", sample_input.shape)
    print("Output Shape:", output.shape)
