import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from timm import create_model
import torch.nn.functional as F


class VisionTransformerLSTMv1(nn.Module):
    def __init__(
        self,
        num_classes=2,
        model_name='vit_base_patch16_224',
        use_temporal_modeling=False,
        temporal_hidden_size=128,
        dropout_p=0.5,
        rnn_num_layers=1,
        bidirectional=False,
        freeze_vit=False
    ):
        """
        Initializes the VisionTransformer model with optional temporal modeling and sequence-level masking.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 2.
            use_temporal_modeling (bool, optional): Whether to use temporal modeling (LSTM). Defaults to False.
            temporal_hidden_size (int, optional): Hidden size for the LSTM. Defaults to 128.
            dropout_p (float, optional): Dropout probability. Defaults to 0.5.
            rnn_num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            bidirectional (bool, optional): Whether the LSTM is bidirectional. Defaults to False.
            freeze_vit (bool, optional): Whether to freeze ViT parameters. Defaults to False.
        """
        super(VisionTransformerLSTMv1, self).__init__()
        
        # Initialize the ViT model
        self.vit = create_model(model_name, pretrained=True)  # Replace with your specific model
        
        # Optionally freeze the Vision Transformer parameters
        if freeze_vit:
            print("Freezing Vision Transformer parameters.")
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Extract the number of input features for the classification head
        try:
            in_features = self.vit.head.in_features  # Timm's ViT has 'head'
        except AttributeError:
            in_features = self.vit.heads.in_features  # Adjust based on your model
        
        # Replace the classification head with an identity function to extract features
        self.vit.head = nn.Identity()
        
        self.use_temporal_model = use_temporal_modeling
        if self.use_temporal_model:
            # Initialize the LSTM with the correct input size
            self.temporal_model = nn.LSTM(
                input_size=in_features,  # Match ViT feature size
                hidden_size=temporal_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True, 
                dropout=dropout_p if rnn_num_layers > 1 else 0.0,  # Dropout only if num_layers > 1
                bidirectional=bidirectional,
            )
            lstm_output_size = temporal_hidden_size * (2 if bidirectional else 1)
            self.temporal_fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(lstm_output_size, num_classes)
            )
        else:
            # Define a separate classification head
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x, img_mask=None, seq_mask=None):
        """
        Forward pass of the Vision Transformer model with optional temporal modeling and sequence-level masking.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_frames, channels, height, width].
            mask (Tensor, optional): Mask tensor of shape [batch_size, num_frames], where 1 indicates valid frames and 0 indicates padded frames.

        Returns:
            Tensor: Output logits of shape [batch_size, num_classes].
        """
        # Ensure input has the correct dimensions
        assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D input."

        batch_size, num_frames, c, h, w = x.size()

        # Flatten the batch and frame dimensions to pass through ViT
        x = x.view(batch_size * num_frames, c, h, w)

        # Extract features using ViT
        x = self.vit(x)  # Shape: [batch_size * num_frames, feature_dim]

        # Reshape back to [batch_size, num_frames, feature_dim]
        feature_dim = x.size(-1)
        x = x.view(batch_size, num_frames, feature_dim)

        if self.use_temporal_model:
            # Handle temporal modeling with LSTM
            if seq_mask is None:
                # If no mask is provided, assume all frames are valid
                lengths = torch.full((batch_size,), num_frames, dtype=torch.long, device=x.device)
            else:
                # Compute actual lengths from the mask
                lengths = seq_mask.sum(dim=1).long()  # Shape: [batch_size]

            # Sort the batch by lengths in descending order (required by pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            x_sorted = x[sorted_idx]

            # Pack the sequences for efficient processing
            packed_input = pack_padded_sequence(
                x_sorted, 
                lengths_sorted.cpu(), 
                batch_first=True, 
                enforce_sorted=True
            )

            # Pass through LSTM
            packed_output, (hn, cn) = self.temporal_model(packed_input)

            # Unpack the sequences
            output, _ = pad_packed_sequence(
                packed_output, 
                batch_first=True, 
                total_length=num_frames
            )

            # Restore the original order of the batch
            _, original_idx = sorted_idx.sort()
            output = output[original_idx]

            if seq_mask is not None:
                # Ensure lengths are at least 1 to avoid negative indexing
                lengths = lengths.clamp(min=1)
                # Gather the last valid output for each sequence
                last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, 1, output.size(2))
                last_outputs = output.gather(1, last_indices).squeeze(1)  # Shape: [batch_size, hidden_size]
            else:
                # If no mask, take the last time step
                last_outputs = output[:, -1, :]  # Shape: [batch_size, hidden_size]

            # Pass through the temporal fully connected layer to get logits
            x = self.temporal_fc(last_outputs)  # Shape: [batch_size, num_classes]
        else:
            # Handle simple feature aggregation without temporal modeling
            if seq_mask is not None:
                # Expand mask to match feature dimensions
                mask_expanded = seq_mask.unsqueeze(-1).float()  # Shape: [batch_size, num_frames, 1]
                # Zero out features of padded frames
                x = x * mask_expanded
                # Sum the features over valid frames
                sum_features = x.sum(dim=1)  # Shape: [batch_size, feature_dim]
                # Count the number of valid frames
                counts = mask_expanded.sum(dim=1)  # Shape: [batch_size, 1]
                counts = counts.clamp(min=1)  # Avoid division by zero
                # Compute the mean over valid frames
                x = sum_features / counts
            else:
                # If no mask is provided, compute the mean normally
                x = x.mean(dim=1)  # Shape: [batch_size, feature_dim]

            # Pass through the classification head to get logits
            x = self.classifier(x)  # Shape: [batch_size, num_classes]

        return x

class VisionTransformerLSTM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        model_name='vit_base_patch16_224',
        use_temporal_modeling=True,
        temporal_hidden_size=128,
        dropout_p=0.5,
        rnn_num_layers=1,
        bidirectional=False,
        freeze_backbone=False,
        pooling='avg'  # Options: 'avg', 'max', 'none'
    ):
        super(VisionTransformerLSTM, self).__init__()
        
        self.backbone = create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)
        self.model_name = model_name.lower()

        if freeze_backbone:
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.pooling = pooling.lower()
        
        # Determine model type and feature handling strategy
        self.backbone.eval()
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            try:
                sample_output = self.backbone.forward_features(sample_input)
            except AttributeError:
                sample_output = self.backbone(sample_input)
                print(f"Warning: Model '{model_name}' does not have 'forward_features'. Using 'forward' instead.")
            
            # Detect model type and output format
            if any(name in self.model_name for name in ['vit', 'deit']):
                self.model_type = 'vit'
                self.feature_dim = sample_output.size(-1)
                print(f"Detected ViT-like model with feature dim {self.feature_dim}")
            elif 'swin' in self.model_name:
                self.model_type = 'swin'
                if sample_output.dim() == 4:
                    # Some Swin variants output feature maps
                    self.feature_dim = sample_output.size(1)  # Channel dimension
                else:
                    # Some output flattened features
                    self.feature_dim = sample_output.size(-1)
                print(f"Detected Swin Transformer with feature dim {self.feature_dim}")
            elif 'convnext' in self.model_name:
                self.model_type = 'convnext'
                if sample_output.dim() == 4:
                    # ConvNeXt typically outputs feature maps
                    self.feature_dim = sample_output.size(1)  # Channel dimension
                else:
                    self.feature_dim = sample_output.size(-1)
                print(f"Detected ConvNeXt with feature dim {self.feature_dim}")
            else:
                # Generic handling for other architectures
                self.model_type = 'generic'
                if sample_output.dim() == 2:
                    self.feature_dim = sample_output.size(-1)
                elif sample_output.dim() == 3:
                    self.feature_dim = sample_output.size(-1)
                elif sample_output.dim() == 4:
                    self.feature_dim = sample_output.size(1)  # Channel dimension
                else:
                    raise ValueError(f"Unexpected output shape: {sample_output.shape}")
                print(f"Using generic handling for model with feature dim {self.feature_dim}")

        print(f"Model type: {self.model_type}")
        print(f"Output shape from backbone: {sample_output.shape}")
        self.backbone.train()

        # Projection layer
        if self.model_type in ['convnext', 'generic'] and sample_output.dim() == 4:
            # For models outputting feature maps, add pooling before projection
            if self.pooling == 'avg':
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
            elif self.pooling == 'max':
                self.pool = nn.AdaptiveMaxPool2d((1, 1))
            else:
                self.pool = nn.Identity()
                self.feature_dim = sample_output.size(1) * sample_output.size(2) * sample_output.size(3)

        self.proj = nn.Linear(self.feature_dim, temporal_hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, temporal_hidden_size))
        
        # LSTM setup
        self.use_temporal_model = use_temporal_modeling
        if self.use_temporal_model:
            self.lstm = nn.LSTM(
                input_size=temporal_hidden_size,
                hidden_size=temporal_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=dropout_p if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
            lstm_output_size = temporal_hidden_size * (2 if bidirectional else 1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(lstm_output_size, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(temporal_hidden_size, num_classes)
            )
        
    def extract_features(self, x):
        """Extract features based on model type"""
        try:
            features = self.backbone.forward_features(x)
        except AttributeError:
            features = self.backbone(x)
            
        if self.model_type == 'vit':
            # For ViT/DeiT, take CLS token or average all tokens
            if features.dim() == 3:
                features = features[:, 0]  # Take CLS token
        elif self.model_type == 'swin':
            if features.dim() == 4:
                # Proper pooling for Swin Transformer output
                if self.pooling == 'avg':
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                elif self.pooling == 'max':
                    features = F.adaptive_max_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1) # Flatten to (batch_size, channels)
        elif self.model_type == 'convnext':
            if features.dim() == 4:
                if hasattr(self, 'pool'):
                    features = self.pool(features)
                features = features.flatten(1)
        else:  # generic
            if features.dim() == 4:
                if hasattr(self, 'pool'):
                    features = self.pool(features)
                features = features.flatten(1)
            elif features.dim() == 3:
                features = features.mean(dim=1)  # Average across sequence dimension

        return features
        
    def forward(self, x, img_mask=None, seq_mask=None):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)

        # Extract features using appropriate method for model type
        features = self.extract_features(x)
        
        # Project features
        features = self.proj(features)
        features = features.view(batch_size, num_frames, -1)
        
        # Add positional encoding
        features = features + self.positional_encoding.expand(batch_size, num_frames, -1)
        
        if self.use_temporal_model:
            # Handle sequence masking
            if seq_mask is not None:
                lengths = seq_mask.sum(dim=1).long()
                lengths_sorted, sorted_idx = lengths.sort(descending=True)
                features_sorted = features[sorted_idx]
                
                packed_input = pack_padded_sequence(
                    features_sorted,
                    lengths_sorted.cpu(),
                    batch_first=True,
                    enforce_sorted=True
                )
                
                packed_output, _ = self.lstm(packed_input)
                output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=num_frames)
                
                _, original_idx = sorted_idx.sort()
                output = output[original_idx]
                
                lengths = lengths.clamp(min=1)
                last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, 1, output.size(2))
                last_outputs = output.gather(1, last_indices).squeeze(1)
            else:
                output, _ = self.lstm(features)
                last_outputs = output[:, -1, :]
            
            logits = self.classifier(last_outputs)
        else:
            if seq_mask is not None:
                mask_expanded = seq_mask.unsqueeze(-1).float()
                features = features * mask_expanded
                aggregated = features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                aggregated = features.mean(dim=1)
            
            logits = self.classifier(aggregated)
        
        return logits


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        model_name='resnet50',
        num_classes=1000,
        num_frames=16,
        num_channels=[512, 256, 128],
        kernel_size=3,
        dropout=0.1,
        pooling='avg'  # Options: 'avg', 'max', 'none'
    ):
        super(TemporalConvNet, self).__init__()
        
        # Initialize backbone
        self.backbone = create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)
        self.model_name = model_name.lower()
        self.num_frames = num_frames
        self.pooling = pooling.lower()
        
        # Determine model type and feature handling strategy
        self.backbone.eval()
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            try:
                sample_output = self.backbone.forward_features(sample_input)
            except AttributeError:
                sample_output = self.backbone(sample_input)
                print(f"Warning: Model '{model_name}' does not have 'forward_features'. Using 'forward' instead.")
            
            # Detect model type and output format
            if any(name in self.model_name for name in ['vit', 'deit']):
                self.model_type = 'vit'
                self.feature_dim = sample_output.size(-1)
                print(f"Detected ViT-like model with feature dim {self.feature_dim}")
            elif 'swin' in self.model_name:
                self.model_type = 'swin'
                if sample_output.dim() == 4:
                    self.feature_dim = sample_output.size(1)
                else:
                    self.feature_dim = sample_output.size(-1)
                print(f"Detected Swin Transformer with feature dim {self.feature_dim}")
            elif 'convnext' in self.model_name:
                self.model_type = 'convnext'
                if sample_output.dim() == 4:
                    self.feature_dim = sample_output.size(1)
                else:
                    self.feature_dim = sample_output.size(-1)
                print(f"Detected ConvNeXt with feature dim {self.feature_dim}")
            else:
                self.model_type = 'generic'
                if sample_output.dim() == 2:
                    self.feature_dim = sample_output.size(-1)
                elif sample_output.dim() == 3:
                    self.feature_dim = sample_output.size(-1)
                elif sample_output.dim() == 4:
                    self.feature_dim = sample_output.size(1)
                else:
                    raise ValueError(f"Unexpected output shape: {sample_output.shape}")
                print(f"Using generic handling for model with feature dim {self.feature_dim}")

        print(f"Model type: {self.model_type}")
        print(f"Output shape from backbone: {sample_output.shape}")
        self.backbone.train()

        # Add pooling layers for feature map outputs
        if sample_output.dim() == 4:
            if self.pooling == 'avg':
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
            elif self.pooling == 'max':
                self.pool = nn.AdaptiveMaxPool2d((1, 1))
            else:
                self.pool = nn.Identity()
                self.feature_dim = sample_output.size(1) * sample_output.size(2) * sample_output.size(3)
        
        # Temporal Convolutional Layers
        layers = []
        in_channels = self.feature_dim
        for out_channels in num_channels:
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size//2)),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ]
            in_channels = out_channels
        self.temporal_conv = nn.Sequential(*layers)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Head
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def extract_features(self, x):
        """Extract features based on model type"""
        try:
            features = self.backbone.forward_features(x)
        except AttributeError:
            features = self.backbone(x)
            
        if self.model_type == 'vit':
            # For ViT/DeiT, take CLS token or average all tokens
            if features.dim() == 3:
                features = features[:, 0]  # Take CLS token
        elif self.model_type == 'swin':
            if features.dim() == 4:
                if hasattr(self, 'pool'):
                    features = self.pool(features)
                features = features.flatten(1)
        elif self.model_type == 'convnext':
            if features.dim() == 4:
                if hasattr(self, 'pool'):
                    features = self.pool(features)
                features = features.flatten(1)
        else:  # generic
            if features.dim() == 4:
                if hasattr(self, 'pool'):
                    features = self.pool(features)
                features = features.flatten(1)
            elif features.dim() == 3:
                features = features.mean(dim=1)  # Average across sequence dimension
                
        return features
    
    def forward(self, x, img_mask=None, seq_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, C, H, W)
            img_mask: Optional mask for images (not used in this model)
            seq_mask: Optional mask for sequence (not used in this model)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extract features using appropriate method for model type
        features = self.extract_features(x)
        
        # Reshape features for temporal processing
        features = features.view(batch_size, num_frames, -1)
        
        # Transpose for Conv1d: (batch_size, feature_dim, num_frames)
        features = features.permute(0, 2, 1)
        
        # Apply temporal convolutions
        features = self.temporal_conv(features)
        
        # Apply sequence mask if provided (optional)
        if seq_mask is not None:
            mask = seq_mask.unsqueeze(1)  # Add channel dimension
            features = features * mask
        
        # Global Average Pooling
        features = self.global_pool(features).squeeze(-1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits

    
class TransformerTemporalModel(nn.Module):
    def __init__(
        self,
        model_name='vit_base_patch16_224',
        num_classes=2,
        num_frames=16,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        pooling='avg'  # Options: 'avg', 'max', 'none'
    ):
        super(TransformerTemporalModel, self).__init__()
        
        # Initialize backbone
        self.backbone = create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)
        self.model_name = model_name.lower()
        self.pooling = pooling.lower()
        
        # Determine model type and feature dimension
        self.backbone.eval()
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            try:
                sample_output = self.backbone.forward_features(sample_input)
            except AttributeError:
                sample_output = self.backbone(sample_input)
                print(f"Warning: Model '{model_name}' does not have 'forward_features'. Using 'forward' instead.")
            
            # Detect model type and set feature dimension
            if any(name in self.model_name for name in ['vit', 'deit']):
                self.model_type = 'vit'
                self.feature_dim = sample_output.size(-1)
            elif 'swin' in self.model_name:
                self.model_type = 'swin'
                self.feature_dim = sample_output.size(-1) if sample_output.dim() == 2 else sample_output.size(1)
            elif 'convnext' in self.model_name:
                self.model_type = 'convnext'
                self.feature_dim = sample_output.size(-1) if sample_output.dim() == 2 else sample_output.size(1)
            else:
                self.model_type = 'generic'
                self.feature_dim = (sample_output.size(-1) if sample_output.dim() <= 3 
                                  else sample_output.size(1))
            
            print(f"Model type: {self.model_type}")
            print(f"Feature dimension: {self.feature_dim}")
            print(f"Output shape from backbone: {sample_output.shape}")
        
        self.backbone.train()
        
        # Add pooling for feature map outputs
        if sample_output.dim() == 4:
            if self.pooling == 'avg':
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
            elif self.pooling == 'max':
                self.pool = nn.AdaptiveMaxPool2d((1, 1))
            else:
                self.pool = nn.Identity()
                self.feature_dim = sample_output.size(1) * sample_output.size(2) * sample_output.size(3)
        
        # Projection layer to match transformer dimension
        self.proj = nn.Linear(self.feature_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def extract_features(self, x):
        """Extract features based on model type"""
        try:
            features = self.backbone.forward_features(x)
        except AttributeError:
            features = self.backbone(x)
        
        if self.model_type == 'vit':
            if features.dim() == 3:
                features = features[:, 0]  # Take CLS token
        elif features.dim() == 4:
            if hasattr(self, 'pool'):
                features = self.pool(features)
            features = features.flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)
        
        return features
    
    def forward(self, x, img_mask=None, seq_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, num_frames, C, H, W)
            img_mask: Optional mask for images (not used)
            seq_mask: Optional mask for transformer sequence
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Process each frame through the backbone
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.extract_features(x)
        
        # Project and reshape features
        features = self.proj(features)
        features = features.view(batch_size, num_frames, -1)
        
        # Add positional encoding
        features = features + self.pos_encoding
        
        # Create attention mask if seq_mask is provided
        if seq_mask is not None:
            # Create attention mask (1 for valid positions, 0 for padding)
            attn_mask = seq_mask.logical_not()
        else:
            attn_mask = None
        
        # Pass through transformer
        features = self.transformer(features, src_key_padding_mask=attn_mask)
        
        # Global pooling over temporal dimension
        if seq_mask is not None:
            # Masked mean
            mask = seq_mask.float().unsqueeze(-1)
            features = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            # Regular mean
            features = features.mean(dim=1)
        
        # Normalize and classify
        features = self.norm(features)
        logits = self.classifier(features)
        
        return logits


# class LoRALinear(nn.Module):
#     def __init__(
#         self, 
#         original_linear: nn.Linear, 
#         r: int = 4, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.1, 
#         merge_weights: bool = False
#     ):
#         """
#         LoRA module for a linear layer.

#         Args:
#             original_linear (nn.Linear): The original linear layer to be adapted.
#             r (int, optional): Rank of the LoRA matrices. Defaults to 4.
#             lora_alpha (int, optional): Scaling factor. Defaults to 1.
#             lora_dropout (float, optional): Dropout probability. Defaults to 0.1.
#             merge_weights (bool, optional): If True, merge LoRA weights into the original layer after training. Defaults to False.
#         """
#         super(LoRALinear, self).__init__()
#         self.original_linear = original_linear
#         self.r = r
#         self.lora_alpha = lora_alpha
#         self.lora_dropout = nn.Dropout(p=lora_dropout)
#         self.merge_weights = merge_weights

#         if r > 0:
#             # Initialize A and B matrices for LoRA
#             self.lora_A = nn.Parameter(torch.zeros((r, original_linear.in_features)))
#             self.lora_B = nn.Parameter(torch.zeros((original_linear.out_features, r)))
#             # Initialize A and B
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)
#             # Scaling
#             self.scaling = self.lora_alpha / self.r
#         else:
#             self.lora_A = None
#             self.lora_B = None
#             self.scaling = 1

#     def forward(self, x):
#         if self.r > 0:
#             # Apply dropout to the input (optional, based on lora_dropout)
#             x = self.lora_dropout(x)
#             # Compute LoRA contribution: (x @ A^T) @ B^T
#             lora_out = F.linear(x, self.lora_A)  # Shape: [batch_size, r]
#             lora_out = F.linear(lora_out, self.lora_B)  # Shape: [batch_size, out_features]
#             # Apply scaling
#             lora_out = lora_out * self.scaling
#             return self.original_linear(x) + lora_out
#         else:
#             return self.original_linear(x)

#     def merge(self):
#         """
#         Merge LoRA weights into the original linear layer. This is useful for inference to reduce model size.
#         """
#         if self.r > 0 and not self.merge_weights:
#             # Compute the merged weight
#             merged_weight = self.original_linear.weight.data + (self.lora_B @ self.lora_A) * self.scaling
#             self.original_linear.weight.data = merged_weight
#             # Optionally, remove LoRA parameters
#             self.lora_A = None
#             self.lora_B = None
#             self.r = 0
#             self.merge_weights = True

# def apply_lora_to_vit(model: nn.Module, r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.1):
#     """
#     Recursively replace linear layers in the ViT model with LoRALinear.

#     Args:
#         model (nn.Module): The Vision Transformer model.
#         r (int, optional): Rank of the LoRA matrices. Defaults to 4.
#         lora_alpha (int, optional): Scaling factor. Defaults to 1.
#         lora_dropout (float, optional): Dropout probability. Defaults to 0.1.
#     """
#     for name, module in model.named_children():
#         if isinstance(module, nn.Linear):
#             print(f"Applying LoRA to layer: {name}")
#             # Replace with LoRALinear
#             lora_module = LoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
#             setattr(model, name, lora_module)
#         else:
#             apply_lora_to_vit(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

# def merge_lora_weights(model: nn.Module):
#     """
#     Recursively merge LoRA weights into the original linear layers.

#     Args:
#         model (nn.Module): The model containing LoRALinear layers.
#     """
#     for module in model.modules():
#         if isinstance(module, LoRALinear):
#             module.merge()


# class VisionTransformerWithLoRA(nn.Module):
#     def __init__(
#         self,
#         num_classes=2,
#         model_name='vit_base_patch16_224',
#         use_temporal_modeling=False,
#         temporal_hidden_size=128,
#         dropout_p=0.5,
#         rnn_num_layers=1,
#         bidirectional=False,
#         freeze_vit=False,
#         use_lora=False,            # New parameter to enable LoRA
#         lora_r=4,                  # LoRA rank
#         lora_alpha=1,              # LoRA scaling factor
#         lora_dropout=0.1           # LoRA dropout probability
#     ):
#         """
#         Initializes the VisionTransformerWithLoRA model with optional temporal modeling, LoRA finetuning, and sequence-level masking.

#         Args:
#             num_classes (int, optional): Number of output classes. Defaults to 2.
#             model_name (str, optional): Name of the ViT model to use from timm. Defaults to 'vit_base_patch16_224'.
#             use_temporal_modeling (bool, optional): Whether to use temporal modeling (LSTM). Defaults to False.
#             temporal_hidden_size (int, optional): Hidden size for the LSTM. Defaults to 128.
#             dropout_p (float, optional): Dropout probability. Defaults to 0.5.
#             rnn_num_layers (int, optional): Number of LSTM layers. Defaults to 1.
#             bidirectional (bool, optional): Whether the LSTM is bidirectional. Defaults to False.
#             freeze_vit (bool, optional): Whether to freeze ViT parameters. Defaults to False.
#             use_lora (bool, optional): Whether to apply LoRA to ViT layers. Defaults to False.
#             lora_r (int, optional): LoRA rank. Defaults to 4.
#             lora_alpha (int, optional): LoRA scaling factor. Defaults to 1.
#             lora_dropout (float, optional): LoRA dropout probability. Defaults to 0.1.
#         """
#         super(VisionTransformerWithLoRA, self).__init__()
        
#         # Initialize the ViT model
#         self.vit = create_model(model_name, pretrained=True)
        
#         # Optionally apply LoRA to the ViT model
#         if use_lora:
#             print("Applying LoRA to Vision Transformer layers.")
#             apply_lora_to_vit(self.vit, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
#         # Optionally freeze the Vision Transformer parameters (excluding LoRA parameters)
#         if freeze_vit:
#             print("Freezing Vision Transformer parameters (excluding LoRA parameters).")
#             for name, param in self.vit.named_parameters():
#                 if not any(layer in name.lower() for layer in ['lora_a', 'lora_b']):
#                     param.requires_grad = False
        
#         # Extract the number of input features for the classification head
#         try:
#             in_features = self.vit.head.in_features  # Timm's ViT has 'head'
#         except AttributeError:
#             in_features = self.vit.heads.in_features  # Adjust based on your model
        
#         # Replace the classification head with an identity function to extract features
#         self.vit.head = nn.Identity()
        
#         self.use_temporal_model = use_temporal_modeling
#         if self.use_temporal_model:
#             # Initialize the LSTM with the correct input size
#             self.temporal_model = nn.LSTM(
#                 input_size=in_features,  # Match ViT feature size
#                 hidden_size=temporal_hidden_size,
#                 num_layers=rnn_num_layers,
#                 batch_first=True, 
#                 dropout=dropout_p if rnn_num_layers > 1 else 0.0,  # Dropout only if num_layers > 1
#                 bidirectional=bidirectional,
#             )
#             lstm_output_size = temporal_hidden_size * (2 if bidirectional else 1)
#             self.temporal_fc = nn.Sequential(
#                 nn.Dropout(p=dropout_p),
#                 nn.Linear(lstm_output_size, num_classes)
#             )
#         else:
#             # Define a separate classification head
#             self.classifier = nn.Sequential(
#                 nn.Dropout(p=dropout_p),
#                 nn.Linear(in_features, num_classes)
#             )
    
#     def forward(self, x, img_mask=None, seq_mask=None):
#         """
#         Forward pass of the Vision Transformer model with optional temporal modeling and sequence-level masking.

#         Args:
#             x (Tensor): Input tensor of shape [batch_size, num_frames, channels, height, width].
#             img_mask (Tensor, optional): Image mask tensor (unused in this implementation).
#             seq_mask (Tensor, optional): Mask tensor of shape [batch_size, num_frames], where 1 indicates valid frames and 0 indicates padded frames.

#         Returns:
#             Tensor: Output logits of shape [batch_size, num_classes].
#         """
#         # Ensure input has the correct dimensions
#         assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D input."

#         batch_size, num_frames, c, h, w = x.size()

#         # Flatten the batch and frame dimensions to pass through ViT
#         x = x.view(batch_size * num_frames, c, h, w)

#         # Extract features using ViT
#         x = self.vit(x)  # Shape: [batch_size * num_frames, feature_dim]

#         # Reshape back to [batch_size, num_frames, feature_dim]
#         feature_dim = x.size(-1)
#         x = x.view(batch_size, num_frames, feature_dim)

#         if self.use_temporal_model:
#             # Handle temporal modeling with LSTM
#             if seq_mask is None:
#                 # If no mask is provided, assume all frames are valid
#                 lengths = torch.full((batch_size,), num_frames, dtype=torch.long, device=x.device)
#             else:
#                 # Compute actual lengths from the mask
#                 lengths = seq_mask.sum(dim=1).long()  # Shape: [batch_size]

#             # Sort the batch by lengths in descending order (required by pack_padded_sequence)
#             lengths_sorted, sorted_idx = lengths.sort(descending=True)
#             x_sorted = x[sorted_idx]

#             # Pack the sequences for efficient processing
#             packed_input = pack_padded_sequence(
#                 x_sorted, 
#                 lengths_sorted.cpu(), 
#                 batch_first=True, 
#                 enforce_sorted=True
#             )

#             # Pass through LSTM
#             packed_output, (hn, cn) = self.temporal_model(packed_input)

#             # Unpack the sequences
#             output, _ = pad_packed_sequence(
#                 packed_output, 
#                 batch_first=True, 
#                 total_length=num_frames
#             )

#             # Restore the original order of the batch
#             _, original_idx = sorted_idx.sort()
#             output = output[original_idx]

#             if seq_mask is not None:
#                 # Ensure lengths are at least 1 to avoid negative indexing
#                 lengths = lengths.clamp(min=1)
#                 # Gather the last valid output for each sequence
#                 last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, 1, output.size(2))
#                 last_outputs = output.gather(1, last_indices).squeeze(1)  # Shape: [batch_size, hidden_size]
#             else:
#                 # If no mask, take the last time step
#                 last_outputs = output[:, -1, :]  # Shape: [batch_size, hidden_size]

#             # Pass through the temporal fully connected layer to get logits
#             x = self.temporal_fc(last_outputs)  # Shape: [batch_size, num_classes]
#         else:
#             # Handle simple feature aggregation without temporal modeling
#             if seq_mask is not None:
#                 # Expand mask to match feature dimensions
#                 mask_expanded = seq_mask.unsqueeze(-1).float()  # Shape: [batch_size, num_frames, 1]
#                 # Zero out features of padded frames
#                 x = x * mask_expanded
#                 # Sum the features over valid frames
#                 sum_features = x.sum(dim=1)  # Shape: [batch_size, feature_dim]
#                 # Count the number of valid frames
#                 counts = mask_expanded.sum(dim=1)  # Shape: [batch_size, 1]
#                 counts = counts.clamp(min=1)  # Avoid division by zero
#                 # Compute the mean over valid frames
#                 x = sum_features / counts
#             else:
#                 # If no mask is provided, compute the mean normally
#                 x = x.mean(dim=1)  # Shape: [batch_size, feature_dim]

#             # Pass through the classification head to get logits
#             x = self.classifier(x)  # Shape: [batch_size, num_classes]

#         return x