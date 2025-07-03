"""
@File    : dan.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: Dynamic Agent Network (DAN) implementation for handling dynamic agent numbers and observation dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DAN(nn.Module):
    """Dynamic Agent Network (DAN) for handling dynamic agent numbers and observation dimensions
    
    Based on the paper's description, DAN uses dual encoders:
    1. Environmental encoder: processes environmental information
    2. Agent interaction encoder: processes neighboring agent information
    3. Attention mechanism: aggregates features from neighboring agents
    """
    
    def __init__(self, env_obs_dim, neighbor_obs_dim, hidden_dim=128, num_heads=4, 
                 use_attention=True, dropout=0.1, layer_norm=True):
        super(DAN, self).__init__()
        
        self.env_obs_dim = env_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim  # Full dimension of neighbor observations
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.layer_norm = layer_norm
        
        # Environmental information encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(env_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU()
        )
        
        # Agent interaction information encoder - processes full neighbor observations
        self.agent_encoder = nn.Sequential(
            nn.Linear(neighbor_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU()
        )
        
        # Multi-head attention for neighboring agents
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            if layer_norm:
                self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, env_obs, neighbor_obs, agent_mask=None):
        """Forward pass of DAN
        Args:
            env_obs: Environmental observations [batch_size, env_obs_dim]
            neighbor_obs: Neighboring agent observations [batch_size, num_neighbors, neighbor_obs_dim]
            agent_mask: Mask for valid agents [batch_size, num_neighbors] (1 for valid, 0 for invalid)
        Returns:
            encoded_features: Encoded features [batch_size, hidden_dim]
            attention_weights: Attention weights if using attention [batch_size, num_heads, 1, num_neighbors]
        """
        batch_size = env_obs.size(0)
        attention_weights = None
        
        # Encode environmental information
        env_features = self.env_encoder(env_obs)  # [batch_size, hidden_dim]
        
        # Handle neighbor observations
        if neighbor_obs.dim() == 2:
            # Single neighbor case: [batch_size, neighbor_obs_dim]
            neighbor_obs = neighbor_obs.unsqueeze(1)  # [batch_size, 1, neighbor_obs_dim]
            if agent_mask is not None:
                agent_mask = agent_mask.unsqueeze(1)  # [batch_size, 1]
                
        # Encode agent interaction information
        num_neighbors = neighbor_obs.size(1)
        neighbor_obs_dim = neighbor_obs.size(2)
        neighbor_features = self.agent_encoder(
            neighbor_obs.view(-1, neighbor_obs_dim)
        ).view(batch_size, num_neighbors, self.hidden_dim)
        
        # Aggregate neighboring agent features
        if self.use_attention and num_neighbors > 0:
            # Use attention mechanism
            query = env_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            key = value = neighbor_features  # [batch_size, num_neighbors, hidden_dim]
            
            # Prepare attention mask
            key_padding_mask = None
            if agent_mask is not None:
                # agent_mask: 1 for valid, 0 for invalid
                # key_padding_mask: True for invalid, False for valid
                key_padding_mask = (agent_mask == 0)
                
            attended_features, attention_weights = self.attention(
                query, key, value, key_padding_mask=key_padding_mask
            )
            agent_features = attended_features.squeeze(1)  # [batch_size, hidden_dim]
            
            # Apply layer normalization if enabled
            if self.layer_norm:
                agent_features = self.attention_norm(agent_features)
        else:
            # Use average pooling
            if agent_mask is not None and num_neighbors > 0:
                # Mask invalid agents
                masked_features = neighbor_features * agent_mask.unsqueeze(-1)
                valid_count = agent_mask.sum(dim=1, keepdim=True).clamp(min=1)
                agent_features = masked_features.sum(dim=1) / valid_count
            elif num_neighbors > 0:
                agent_features = neighbor_features.mean(dim=1)
            else:
                # No neighbors, use zero features
                agent_features = torch.zeros_like(env_features)
                
        # Fuse environmental and agent features
        fused_features = torch.cat([env_features, agent_features], dim=-1)
        encoded_features = self.fusion(fused_features)
        
        # Final output projection
        output_features = self.output_proj(encoded_features)
        
        return output_features, attention_weights
        
    def get_attention_weights(self, env_obs, neighbor_obs, agent_mask=None):
        """Get attention weights for visualization
        Args:
            env_obs: Environmental observations
            neighbor_obs: Neighboring agent observations
            agent_mask: Mask for valid agents
        Returns:
            attention_weights: Attention weights [batch_size, num_heads, 1, num_neighbors]
        """
        with torch.no_grad():
            _, attention_weights = self.forward(env_obs, neighbor_obs, agent_mask)
        return attention_weights
        
    def encode_env_only(self, env_obs):
        """Encode only environmental information
        Args:
            env_obs: Environmental observations [batch_size, env_obs_dim]
        Returns:
            env_features: Encoded environmental features [batch_size, hidden_dim]
        """
        return self.env_encoder(env_obs)
        
    def encode_agent_only(self, agent_obs):
        """Encode only agent information
        Args:
            agent_obs: Agent observations [batch_size, neighbor_obs_dim] or [batch_size, num_agents, neighbor_obs_dim]
        Returns:
            agent_features: Encoded agent features
        """
        if agent_obs.dim() == 3:
            batch_size, num_agents, neighbor_obs_dim = agent_obs.shape
            return self.agent_encoder(
                agent_obs.view(-1, neighbor_obs_dim)
            ).view(batch_size, num_agents, self.hidden_dim)
        else:
            return self.agent_encoder(agent_obs)