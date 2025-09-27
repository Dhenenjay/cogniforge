"""
Policy networks for reinforcement learning and imitation learning.

This module provides policy implementations including simple MLPs,
recurrent policies, and more advanced architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, Any, Optional, Tuple, Union, List
import logging

# Configure logging
logger = logging.getLogger(__name__)


class SimplePolicy(nn.Module):
    """
    Simple 2-layer MLP policy with 64 hidden units per layer.
    
    This policy can be used for continuous or discrete action spaces.
    For continuous actions, it outputs mean and log_std for a Gaussian distribution.
    For discrete actions, it outputs logits for a categorical distribution.
    
    Architecture:
        Input (obs_dim) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Output (act_dim)
        
    For continuous actions:
        - Output is split into mean and log_std
        - Actions are sampled from Normal(mean, exp(log_std))
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        activation: str = 'relu',
        continuous: bool = True,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        deterministic_eval: bool = False
    ):
        """
        Initialize SimplePolicy.
        
        Args:
            obs_dim: Dimension of observation space
            act_dim: Dimension of action space
            hidden_dim: Number of hidden units per layer (default: 64)
            n_hidden_layers: Number of hidden layers (default: 2)
            activation: Activation function ('relu', 'tanh', 'elu')
            continuous: Whether action space is continuous
            log_std_init: Initial value for log standard deviation
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            deterministic_eval: Whether to use deterministic actions during evaluation
        """
        super(SimplePolicy, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.continuous = continuous
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.deterministic_eval = deterministic_eval
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh
        elif activation == 'elu':
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(self.activation())
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation())
        
        # Store as sequential
        self.mlp = nn.Sequential(*layers)
        
        # Output layers
        if continuous:
            # For continuous actions: output mean and log_std
            self.mean_layer = nn.Linear(hidden_dim, act_dim)
            
            # Log std can be either state-dependent or fixed
            self.log_std_type = 'state_dependent'  # or 'fixed'
            
            if self.log_std_type == 'state_dependent':
                self.log_std_layer = nn.Linear(hidden_dim, act_dim)
            else:
                # Fixed log_std as learnable parameter
                self.log_std = nn.Parameter(torch.ones(1, act_dim) * log_std_init)
        else:
            # For discrete actions: output logits
            self.logits_layer = nn.Linear(hidden_dim, act_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Training mode flag
        self.training = True
    
    def _init_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Special initialization for output layers
        if self.continuous:
            # Smaller initial weights for mean layer
            nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(self.mean_layer.bias)
            
            if self.log_std_type == 'state_dependent':
                nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
                nn.init.zeros_(self.log_std_layer.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            deterministic: Whether to return deterministic actions
                          If None, uses self.training to decide
                          
        Returns:
            actions: Action tensor of shape (batch_size, act_dim)
            info: Dictionary with additional information (log_prob, entropy, etc.)
        """
        # Determine whether to use deterministic actions
        if deterministic is None:
            deterministic = not self.training or self.deterministic_eval
        
        # Pass through MLP
        features = self.mlp(obs)
        
        if self.continuous:
            # Continuous action space
            mean = self.mean_layer(features)
            
            # Get log_std
            if self.log_std_type == 'state_dependent':
                log_std = self.log_std_layer(features)
            else:
                log_std = self.log_std.expand_as(mean)
            
            # Clamp log_std
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            
            # Create distribution
            dist = Normal(mean, std)
            
            if deterministic:
                actions = mean
                log_prob = dist.log_prob(actions).sum(dim=-1)
            else:
                # Sample actions
                actions = dist.rsample()  # reparameterization trick
                log_prob = dist.log_prob(actions).sum(dim=-1)
            
            # Compute entropy
            entropy = dist.entropy().sum(dim=-1)
            
            info = {
                'mean': mean,
                'std': std,
                'log_std': log_std,
                'log_prob': log_prob,
                'entropy': entropy,
                'distribution': dist
            }
            
        else:
            # Discrete action space
            logits = self.logits_layer(features)
            
            # Create distribution
            dist = Categorical(logits=logits)
            
            if deterministic:
                # Choose most likely action
                actions = torch.argmax(logits, dim=-1)
            else:
                # Sample actions
                actions = dist.sample()
            
            # Get log probabilities
            log_prob = dist.log_prob(actions)
            
            # Compute entropy
            entropy = dist.entropy()
            
            info = {
                'logits': logits,
                'probs': F.softmax(logits, dim=-1),
                'log_prob': log_prob,
                'entropy': entropy,
                'distribution': dist
            }
        
        return actions, info
    
    def get_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get action for a single observation.
        
        Args:
            obs: Single observation (not batched)
            deterministic: Whether to return deterministic action
            
        Returns:
            action: Single action
        """
        # Handle numpy arrays
        return_numpy = isinstance(obs, np.ndarray)
        if return_numpy:
            obs = torch.FloatTensor(obs)
        
        # Add batch dimension if needed
        single_obs = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single_obs = True
        
        # Get action
        with torch.no_grad():
            actions, _ = self.forward(obs, deterministic=deterministic)
        
        # Remove batch dimension if needed
        if single_obs:
            actions = actions.squeeze(0)
        
        # Convert back to numpy if needed
        if return_numpy:
            actions = actions.cpu().numpy()
        
        return actions
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given obs-action pairs.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
            
        Returns:
            Dictionary with log_prob, entropy, and other info
        """
        # Pass through MLP
        features = self.mlp(obs)
        
        if self.continuous:
            mean = self.mean_layer(features)
            
            if self.log_std_type == 'state_dependent':
                log_std = self.log_std_layer(features)
            else:
                log_std = self.log_std.expand_as(mean)
            
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
        else:
            logits = self.logits_layer(features)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        
        return {
            'log_prob': log_prob,
            'entropy': entropy,
            'distribution': dist
        }
    
    def get_log_prob(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Get log probability of actions given observations.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
            
        Returns:
            Log probability tensor
        """
        return self.evaluate_actions(obs, actions)['log_prob']
    
    def save(self, path: str):
        """
        Save policy to file.
        
        Args:
            path: Path to save file
        """
        torch.save({
            'state_dict': self.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'hidden_dim': self.hidden_dim,
            'n_hidden_layers': self.n_hidden_layers,
            'continuous': self.continuous
        }, path)
        logger.info(f"Saved policy to {path}")
    
    def load(self, path: str):
        """
        Load policy from file.
        
        Args:
            path: Path to saved file
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded policy from {path}")


class RecurrentPolicy(nn.Module):
    """
    Recurrent policy using LSTM or GRU for temporal dependencies.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        recurrent_layers: int = 1,
        recurrent_type: str = 'lstm',
        continuous: bool = True
    ):
        """
        Initialize RecurrentPolicy.
        
        Args:
            obs_dim: Dimension of observation space
            act_dim: Dimension of action space
            hidden_dim: Hidden dimension for recurrent layers
            recurrent_layers: Number of recurrent layers
            recurrent_type: Type of recurrent cell ('lstm' or 'gru')
            continuous: Whether action space is continuous
        """
        super(RecurrentPolicy, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.recurrent_layers = recurrent_layers
        self.recurrent_type = recurrent_type
        self.continuous = continuous
        
        # Input embedding
        self.embed = nn.Linear(obs_dim, hidden_dim)
        
        # Recurrent layers
        if recurrent_type == 'lstm':
            self.recurrent = nn.LSTM(
                hidden_dim, hidden_dim,
                num_layers=recurrent_layers,
                batch_first=True
            )
        elif recurrent_type == 'gru':
            self.recurrent = nn.GRU(
                hidden_dim, hidden_dim,
                num_layers=recurrent_layers,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown recurrent type: {recurrent_type}")
        
        # Output layers
        if continuous:
            self.mean_layer = nn.Linear(hidden_dim, act_dim)
            self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        else:
            self.logits_layer = nn.Linear(hidden_dim, act_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through recurrent policy.
        
        Args:
            obs: Observation tensor (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: Previous hidden state
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions: Action tensor
            hidden: New hidden state
            info: Additional information
        """
        # Handle both sequential and single-step inputs
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
            single_step = True
        else:
            single_step = False
        
        # Embed observations
        embedded = F.relu(self.embed(obs))
        
        # Pass through recurrent layers
        recurrent_out, hidden = self.recurrent(embedded, hidden)
        
        if self.continuous:
            mean = self.mean_layer(recurrent_out)
            log_std = self.log_std_layer(recurrent_out)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            
            if deterministic:
                actions = mean
            else:
                actions = dist.rsample()
            
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            info = {
                'mean': mean,
                'std': std,
                'log_prob': log_prob,
                'entropy': entropy
            }
        else:
            logits = self.logits_layer(recurrent_out)
            dist = Categorical(logits=logits.view(-1, self.act_dim))
            
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample().view(logits.shape[:-1])
            
            log_prob = dist.log_prob(actions.view(-1)).view(actions.shape)
            entropy = dist.entropy().view(actions.shape)
            
            info = {
                'logits': logits,
                'log_prob': log_prob,
                'entropy': entropy
            }
        
        # Remove sequence dimension if single step
        if single_step:
            actions = actions.squeeze(1)
            for key in info:
                if isinstance(info[key], torch.Tensor):
                    info[key] = info[key].squeeze(1)
        
        return actions, hidden, info
    
    def get_initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get initial hidden state.
        
        Args:
            batch_size: Batch size for hidden state
            
        Returns:
            Initial hidden state
        """
        if self.recurrent_type == 'lstm':
            return (
                torch.zeros(self.recurrent_layers, batch_size, self.hidden_dim),
                torch.zeros(self.recurrent_layers, batch_size, self.hidden_dim)
            )
        else:
            return torch.zeros(self.recurrent_layers, batch_size, self.hidden_dim)


class AttentionPolicy(nn.Module):
    """
    Policy with self-attention mechanism for processing variable-length inputs.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        continuous: bool = True,
        max_seq_len: int = 100
    ):
        """
        Initialize AttentionPolicy.
        
        Args:
            obs_dim: Dimension of observation space
            act_dim: Dimension of action space
            hidden_dim: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            continuous: Whether action space is continuous
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(AttentionPolicy, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.continuous = continuous
        
        # Input embedding
        self.embed = nn.Linear(obs_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.1
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Output layers
        if continuous:
            self.mean_layer = nn.Linear(hidden_dim, act_dim)
            self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        else:
            self.logits_layer = nn.Linear(hidden_dim, act_dim)
    
    def forward(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through attention policy.
        
        Args:
            obs: Observation tensor (batch, seq_len, obs_dim)
            mask: Attention mask for padding
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions: Action tensor
            info: Additional information
        """
        batch_size, seq_len = obs.shape[:2]
        
        # Embed and add positional encoding
        embedded = self.embed(obs)
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        features = self.transformer(embedded, src_key_padding_mask=mask)
        
        # Pool over sequence (use last non-masked element or mean)
        if mask is not None:
            # Use last valid element for each sequence
            lengths = (~mask).sum(dim=1)
            pooled = features[torch.arange(batch_size), lengths - 1]
        else:
            # Mean pooling
            pooled = features.mean(dim=1)
        
        if self.continuous:
            mean = self.mean_layer(pooled)
            log_std = self.log_std_layer(pooled)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            
            if deterministic:
                actions = mean
            else:
                actions = dist.rsample()
            
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            info = {
                'mean': mean,
                'std': std,
                'log_prob': log_prob,
                'entropy': entropy,
                'features': features
            }
        else:
            logits = self.logits_layer(pooled)
            dist = Categorical(logits=logits)
            
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample()
            
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            
            info = {
                'logits': logits,
                'log_prob': log_prob,
                'entropy': entropy,
                'features': features
            }
        
        return actions, info


def create_policy(
    policy_type: str,
    obs_dim: int,
    act_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create policies.
    
    Args:
        policy_type: Type of policy ('simple', 'recurrent', 'attention')
        obs_dim: Observation dimension
        act_dim: Action dimension
        **kwargs: Additional arguments for specific policy types
        
    Returns:
        Policy network
        
    Example:
        policy = create_policy('simple', obs_dim=10, act_dim=4, hidden_dim=64)
    """
    if policy_type == 'simple':
        return SimplePolicy(obs_dim, act_dim, **kwargs)
    elif policy_type == 'recurrent':
        return RecurrentPolicy(obs_dim, act_dim, **kwargs)
    elif policy_type == 'attention':
        return AttentionPolicy(obs_dim, act_dim, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


# Example usage
if __name__ == "__main__":
    # Test SimplePolicy
    print("=" * 60)
    print("Testing SimplePolicy")
    print("=" * 60)
    
    # Create policy
    policy = SimplePolicy(obs_dim=10, act_dim=4, continuous=True)
    
    # Test forward pass
    obs = torch.randn(32, 10)  # Batch of 32 observations
    actions, info = policy(obs, deterministic=False)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Mean shape: {info['mean'].shape}")
    print(f"Std shape: {info['std'].shape}")
    print(f"Log prob shape: {info['log_prob'].shape}")
    print(f"Entropy shape: {info['entropy'].shape}")
    
    # Test single action
    single_obs = torch.randn(10)
    single_action = policy.get_action(single_obs)
    print(f"\nSingle observation shape: {single_obs.shape}")
    print(f"Single action shape: {single_action.shape}")
    
    # Test with numpy
    numpy_obs = np.random.randn(10)
    numpy_action = policy.get_action(numpy_obs)
    print(f"\nNumpy observation shape: {numpy_obs.shape}")
    print(f"Numpy action shape: {numpy_action.shape}")
    print(f"Numpy action type: {type(numpy_action)}")
    
    # Test discrete policy
    print("\n" + "=" * 60)
    print("Testing Discrete SimplePolicy")
    print("=" * 60)
    
    discrete_policy = SimplePolicy(obs_dim=10, act_dim=5, continuous=False)
    
    obs = torch.randn(32, 10)
    actions, info = discrete_policy(obs, deterministic=False)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Logits shape: {info['logits'].shape}")
    print(f"Probs shape: {info['probs'].shape}")
    print(f"Log prob shape: {info['log_prob'].shape}")
    
    # Test RecurrentPolicy
    print("\n" + "=" * 60)
    print("Testing RecurrentPolicy")
    print("=" * 60)
    
    recurrent_policy = RecurrentPolicy(obs_dim=10, act_dim=4, continuous=True)
    
    # Single step
    obs = torch.randn(16, 10)
    hidden = recurrent_policy.get_initial_hidden(batch_size=16)
    actions, new_hidden, info = recurrent_policy(obs, hidden)
    
    print(f"Single step:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action shape: {actions.shape}")
    
    # Sequential
    seq_obs = torch.randn(8, 20, 10)  # 8 sequences of length 20
    hidden = recurrent_policy.get_initial_hidden(batch_size=8)
    actions, new_hidden, info = recurrent_policy(seq_obs, hidden)
    
    print(f"\nSequential:")
    print(f"  Observation shape: {seq_obs.shape}")
    print(f"  Action shape: {actions.shape}")
    
    # Test AttentionPolicy
    print("\n" + "=" * 60)
    print("Testing AttentionPolicy")
    print("=" * 60)
    
    attention_policy = AttentionPolicy(obs_dim=10, act_dim=4, continuous=True)
    
    # Variable length sequences
    obs = torch.randn(4, 15, 10)  # 4 sequences of length 15
    mask = torch.zeros(4, 15, dtype=torch.bool)
    mask[0, 10:] = True  # First sequence has length 10
    mask[1, 12:] = True  # Second sequence has length 12
    
    actions, info = attention_policy(obs, mask=mask)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Features shape: {info['features'].shape}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)