import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Initializes the Patch Embedding module.
        
        Args:
            img_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space.
        """
        super(PatchEmbedding, self).__init__()
        
        # Save parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Validate that img_size is divisible by patch_size
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Forward pass of the Patch Embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size]
        
        Returns:
            torch.Tensor: Embedded patches of shape [batch_size, num_patches, embed_dim]
        """
        # Validate input shape
        if x.ndim != 4 or x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            raise ValueError(
                f"Input tensor must have shape [batch_size, {self.proj.in_channels}, {self.img_size}, {self.img_size}], but got {x.shape}"
            )
        
        # Apply Conv2d to project patches into embedding space
        x = self.proj(x)  # [batch_size, embed_dim, num_patches_root, num_patches_root]
        
        # Flatten spatial dimensions into a single dimension for patches
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        
        # Transpose to reorder dimensions: [batch_size, num_patches, embed_dim]
        x = x.transpose(1, 2)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        """
        Initializes the Positional Encoding module.
        
        Args:
            embed_dim (int): Dimension of the embeddings.
            max_len (int, optional): Maximum length of the sequences. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        
        # Check if embed_dim is even
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be even for sinusoidal positional encoding.")
        
        # Initialize positional encoding tensor
        pe = torch.zeros(max_len, embed_dim)
        
        # Compute position indices and divisors
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Populate positional encodings using sine and cosine
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
        
        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embed_dim]
        
        # Register as buffer (non-trainable parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Adds positional encoding to the input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape [batch_size, sequence_length, embed_dim]
        
        Returns:
            torch.Tensor: Positional encoded embeddings.
        """
        # Validate embedding dimension
        if x.size(2) != self.pe.size(2):
            raise ValueError(f"Input embedding dimension ({x.size(2)}) must match positional encoding dimension ({self.pe.size(2)}).")
        
        # Validate sequence length
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Input sequence length ({x.size(1)}) exceeds maximum length ({self.pe.size(1)}). Increase max_len during initialization.")
        
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]  # Broadcast addition
        return x
    
class ClassToken(nn.Module):
    def __init__(self, embed_dim):
        """
        Initializes the Class Token.
        
        Args:
            embed_dim (int): Dimension of the embedding space.
        """
        super(ClassToken, self).__init__()
        # Initialize the class token with zeros and make it trainable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize the class token with a truncated normal distribution
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Prepends the class token to the input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape [batch_size, num_patches, embed_dim]
        
        Returns:
            torch.Tensor: Sequence with class token [batch_size, num_patches + 1, embed_dim]
        """
        # Validate input shape
        if x.size(2) != self.cls_token.size(2):
            raise ValueError(f"Input embedding dimension ({x.size(2)}) must match class token dimension ({self.cls_token.size(2)}).")
        
        # Extract batch size from input tensor
        batch_size = x.size(0)
        
        # Expand the class token to match batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        
        # Concatenate the class token to the beginning of the input sequence
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_dim]
        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Multi-Head Self-Attention module.

        Args:
            embed_dim (int): Dimensionality of embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for query, key, and value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input embeddings of shape [batch_size, seq_len, embed_dim].
            mask (torch.Tensor, optional): Mask of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Output embeddings of shape [batch_size, seq_len, embed_dim].
        """
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding dim ({embed_dim}) must match layer embed_dim ({self.embed_dim})"

        # Compute query, key, and value
        qkv = self.qkv(x)  # [batch_size, seq_len, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into query, key, value

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]

        # # Clamp scores to prevent extremely large values
        # scores = torch.clamp(scores, min=-1e9, max=1e9)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)  # [batch_size, seq_len, embed_dim]

        # Output projection
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        """
        Initializes a single Transformer encoder layer.
        
        Args:
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the feed-forward network.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Layer normalization before attention and MLP
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention sublayer
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network (MLP) with dropout
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),  # First linear layer
            nn.GELU(),  # GELU activation
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(mlp_dim, embed_dim),  # Second linear layer
            nn.Dropout(dropout)  # Final dropout
        )

    def forward(self, x, mask=None):
        """
        Forward pass of the Transformer encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor after the encoder layer.
        """
        # Validate input shape
        assert x.ndim == 3, f"Expected input shape [batch_size, seq_len, embed_dim], but got {x.shape}"
        assert x.size(2) == self.norm1.normalized_shape[0], f"Expected embed_dim={self.norm1.normalized_shape[0]}, but got {x.size(2)}"
        
        # Self-attention sublayer with residual connection
        x = x + self.attn(self.norm1(x), mask=mask)
        
        # Feed-forward sublayer with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dim, dropout=0.1):
        """
        Initializes the Transformer encoder by stacking multiple encoder layers.
        
        Args:
            num_layers (int): Number of encoder layers to stack.
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the feed-forward network.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(TransformerEncoder, self).__init__()
        
        # Ensure at least one layer is defined
        assert num_layers > 0, "num_layers must be greater than 0"
        
        # Create a stack of Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout) 
                for _ in range(num_layers)
            ]
        )
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        Passes the input through each Transformer encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor after the Transformer encoder.
        """
        # Validate input shape
        assert x.ndim == 3, f"Expected input shape [batch_size, seq_len, embed_dim], but got {x.shape}"
        assert x.size(2) == self.layers[0].norm1.normalized_shape[0], \
            f"Input embed_dim ({x.size(2)}) must match encoder embed_dim ({self.layers[0].norm1.normalized_shape[0]})"
        
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        x = self.norm(x)
        return x
    
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=6, mlp_dim=3072, dropout=0.1):
        """
        Initializes the Temporal Transformer Encoder.
        
        Args:
            embed_dim (int, optional): Dimension of the embeddings. Defaults to 768.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            num_layers (int, optional): Number of Transformer encoder layers. Defaults to 6.
            mlp_dim (int, optional): Dimension of the feed-forward network. Defaults to 3072.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(TemporalTransformerEncoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        Forward pass of the Temporal Transformer Encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
            mask (torch.Tensor, optional): Mask tensor of shape [batch_size, seq_len].
        
        Returns:
            torch.Tensor: Output tensor after temporal encoding [batch_size, embed_dim].
        """
        # Validate input shape
        assert x.ndim == 3, f"Expected input shape [batch_size, seq_len, embed_dim], but got {x.shape}"
        assert mask is None or mask.shape == x.shape[:2], \
            f"Mask shape {mask.shape} must match input shape {x.shape[:2]}"

        # Pass through transformer encoder
        x = self.transformer_encoder(x, mask=mask)
        x = self.norm(x)

        # Aggregate sequence information
        if mask is not None:
            x = x * mask.unsqueeze(-1)  # Zero out padded frames
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
            x = x.sum(dim=1) / lengths  # Mean over valid frames
        else:
            x = x.mean(dim=1)  # Mean over all frames
        return x
    
class VisionTransformerWithTemporal(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        use_cls_token=True,
        temporal_num_layers=6,
        temporal_num_heads=12,
        temporal_mlp_dim=3072,
        temporal_dropout=0.1,
    ):
        """
        Vision Transformer with Temporal Modeling.
        
        Args:
            img_size (int): Input image size (assumed square).
            patch_size (int): Size of each image patch (assumed square).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            num_classes (int): Number of output classes for classification.
            embed_dim (int): Embedding dimension for the vision transformer.
            depth (int): Number of transformer encoder layers for the spatial model.
            num_heads (int): Number of attention heads for the spatial model.
            mlp_dim (int): Feed-forward network dimension in the spatial transformer.
            dropout (float): Dropout rate in the spatial transformer.
            emb_dropout (float): Dropout rate after patch embedding.
            use_cls_token (bool): Whether to use a classification token.
            temporal_num_layers (int): Number of transformer encoder layers for the temporal model.
            temporal_num_heads (int): Number of attention heads for the temporal model.
            temporal_mlp_dim (int): Feed-forward network dimension in the temporal transformer.
            temporal_dropout (float): Dropout rate in the temporal transformer.
        """
        super(VisionTransformerWithTemporal, self).__init__()
        
        # Vision Transformer Components
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_patches + 1)
            self.num_tokens = 1
        else:
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_patches)
            self.num_tokens = 0

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer_encoder = TransformerEncoder(
            num_layers=depth, 
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            mlp_dim=mlp_dim, 
            dropout=dropout
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Temporal Transformer Encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            mlp_dim=temporal_mlp_dim,
            dropout=temporal_dropout
        )
        
        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x, img_mask=None, seq_mask=None):
        """
        Forward pass of the Vision Transformer with Temporal Modeling.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, in_channels, img_size, img_size].
            img_mask (torch.Tensor, optional): Mask for valid images [batch_size, seq_len].
            seq_mask (torch.Tensor, optional): Mask for valid sequences [batch_size, seq_len].
        
        Returns:
            torch.Tensor: Classification logits of shape [batch_size, num_classes].
        """
        # Validate input shape
        batch_size, seq_len, in_channels, img_size, _ = x.shape
        assert img_size == self.patch_embed.img_size, \
            f"Input img_size {img_size} does not match expected size {self.patch_embed.img_size}"
        
        # Flatten sequence dimension for image-level processing
        x = x.view(batch_size * seq_len, in_channels, img_size, img_size)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token if enabled
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size * seq_len, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Positional encoding and dropout
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Vision Transformer Encoding
        # if img_mask is not None:
        #     img_mask = img_mask.view(batch_size * seq_len, -1)
        x = self.transformer_encoder(x)
        x = self.norm(x)

        # Check for NaNs after Transformer encoder
        if torch.isnan(x).any():
            raise ValueError("NaN detected after Transformer encoder.")
        
        # Extract features
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        
        # Reshape for temporal processing
        x = x.view(batch_size, seq_len, -1)
        
        # Temporal Transformer Encoding
        x = self.temporal_encoder(x, mask=seq_mask)
        
        # Classification
        logits = self.head(x)
        return logits