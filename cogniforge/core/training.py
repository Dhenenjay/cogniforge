"""
Training utilities for behavior cloning and reinforcement learning.

This module provides training functions for policies including behavior cloning,
fine-tuning, and evaluation utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
import logging
import time
from tqdm import tqdm
import json
from queue import Queue
import threading
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def fit_bc(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_split: float = 0.2,
    early_stopping: bool = True,
    patience: int = 5,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = None,
    scheduler: Optional[str] = None,
    verbose: bool = True,
    device: Optional[str] = None,
    save_best: Optional[str] = None,
    callbacks: Optional[List[Callable]] = None
) -> Dict[str, Any]:
    """
    Fit model using behavior cloning with Adam optimizer and MSE loss.
    
    This function trains a policy network to imitate expert demonstrations using
    supervised learning. It minimizes the MSE between predicted and expert actions.
    
    Args:
        model: Neural network model (e.g., SimplePolicy)
        X: States/observations array of shape (n_samples, state_dim)
        Y: Actions array of shape (n_samples, action_dim)
        epochs: Number of training epochs (default: 20)
        batch_size: Batch size for training (default: 128)
        lr: Learning rate for Adam optimizer (default: 1e-3)
        val_split: Fraction of data to use for validation (default: 0.2)
        early_stopping: Whether to use early stopping (default: True)
        patience: Early stopping patience in epochs (default: 5)
        weight_decay: L2 regularization weight (default: 0.0)
        grad_clip: Gradient clipping value (None = no clipping)
        scheduler: Learning rate scheduler ('cosine', 'step', None)
        verbose: Whether to print training progress (default: True)
        device: Device to use ('cuda', 'cpu', None=auto)
        save_best: Path to save best model (None = don't save)
        callbacks: Optional list of callback functions
        
    Returns:
        Dictionary containing:
        - 'train_losses': List of training losses per epoch
        - 'val_losses': List of validation losses per epoch
        - 'best_val_loss': Best validation loss achieved
        - 'best_epoch': Epoch with best validation loss
        - 'final_train_loss': Final training loss
        - 'final_val_loss': Final validation loss
        - 'training_time': Total training time in seconds
        
    Example:
        from cogniforge.core.policy import SimplePolicy
        from cogniforge.core.training import fit_bc
        
        # Create model
        model = SimplePolicy(obs_dim=10, act_dim=4)
        
        # Train with behavior cloning
        results = fit_bc(model, X_train, Y_train, epochs=50, batch_size=256)
        
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = model.to(device)
    
    # Convert to tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if isinstance(Y, np.ndarray):
        Y = torch.FloatTensor(Y)
    
    # Create dataset and split
    dataset = TensorDataset(X, Y)
    
    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = dataset
        val_dataset = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=(device.type == 'cuda')
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
    else:
        val_loader = None
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    if scheduler == 'cosine':
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
    elif scheduler == 'step':
        scheduler_obj = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.3
        )
    else:
        scheduler_obj = None
    
    # Training state
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Start timer
    start_time = time.time()
    
    if verbose:
        logger.info(f"Starting behavior cloning training")
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val samples")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {batch_size}, Learning rate: {lr}")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_epoch = 0.0
        train_batches = 0
        
        # Use tqdm for progress bar if verbose
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
        
        for batch_idx, (states_batch, actions_batch) in enumerate(train_iter):
            # Move to device
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - get predicted actions
            if hasattr(model, 'forward'):
                # For policies that return (actions, info)
                pred_actions, _ = model(states_batch, deterministic=True)
            else:
                # For simple models
                pred_actions = model(states_batch)
            
            # Compute MSE loss
            loss = F.mse_loss(pred_actions, actions_batch)
            
            # Add L1 loss if specified (for sparsity)
            # loss += 0.001 * torch.mean(torch.abs(pred_actions))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate loss
            train_loss_epoch += loss.item()
            train_batches += 1
            
            # Update progress bar
            if verbose and hasattr(train_iter, 'set_postfix'):
                train_iter.set_postfix({'loss': loss.item()})
        
        # Average training loss
        avg_train_loss = train_loss_epoch / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss_epoch = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for states_batch, actions_batch in val_loader:
                    states_batch = states_batch.to(device)
                    actions_batch = actions_batch.to(device)
                    
                    # Forward pass
                    if hasattr(model, 'forward'):
                        pred_actions, _ = model(states_batch, deterministic=True)
                    else:
                        pred_actions = model(states_batch)
                    
                    # Compute loss
                    loss = F.mse_loss(pred_actions, actions_batch)
                    
                    val_loss_epoch += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_epoch / val_batches
            val_losses.append(avg_val_loss)
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                if save_best:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                    }, save_best)
                    if verbose:
                        logger.info(f"Saved best model to {save_best}")
            else:
                patience_counter += 1
        else:
            avg_val_loss = None
            val_losses.append(None)
        
        # Learning rate scheduling
        if scheduler_obj:
            scheduler_obj.step()
        
        # Logging
        if verbose:
            log_msg = f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                log_msg += f", Val Loss = {avg_val_loss:.6f}"
                if epoch == best_epoch:
                    log_msg += " (best)"
            logger.info(log_msg)
        
        # Callbacks
        if callbacks:
            for callback in callbacks:
                callback(model, epoch, avg_train_loss, avg_val_loss)
        
        # Early stopping
        if early_stopping and patience_counter >= patience:
            if verbose:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    training_time = time.time() - start_time
    
    # Load best model if saved
    if save_best and val_loader:
        checkpoint = torch.load(save_best, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if verbose:
            logger.info(f"Loaded best model from epoch {best_epoch+1}")
    
    # Final summary
    if verbose:
        logger.info(f"Training complete in {training_time:.1f} seconds")
        logger.info(f"Final train loss: {train_losses[-1]:.6f}")
        if val_losses[-1] is not None:
            logger.info(f"Final val loss: {val_losses[-1]:.6f}")
            logger.info(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss if val_loader else train_losses[-1],
        'best_epoch': best_epoch,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1] if val_losses[-1] is not None else None,
        'training_time': training_time,
        'epochs_trained': len(train_losses)
    }


def evaluate_bc(
    model: nn.Module,
    X_test: Union[np.ndarray, torch.Tensor],
    Y_test: Union[np.ndarray, torch.Tensor],
    batch_size: int = 128,
    device: Optional[str] = None,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """
    Evaluate behavior cloning model on test data.
    
    Args:
        model: Trained model
        X_test: Test states
        Y_test: Test actions (ground truth)
        batch_size: Batch size for evaluation
        device: Device to use
        return_predictions: Whether to return predictions
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Convert to tensors
    if isinstance(X_test, np.ndarray):
        X_test = torch.FloatTensor(X_test)
    if isinstance(Y_test, np.ndarray):
        Y_test = torch.FloatTensor(Y_test)
    
    # Create dataloader
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    total_loss = 0.0
    all_predictions = []
    all_errors = []
    n_batches = 0
    
    with torch.no_grad():
        for states_batch, actions_batch in test_loader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            
            # Get predictions
            if hasattr(model, 'forward'):
                pred_actions, _ = model(states_batch, deterministic=True)
            else:
                pred_actions = model(states_batch)
            
            # Compute metrics
            batch_loss = F.mse_loss(pred_actions, actions_batch)
            total_loss += batch_loss.item()
            
            # Store predictions if requested
            if return_predictions:
                all_predictions.append(pred_actions.cpu())
            
            # Compute per-sample errors
            errors = torch.mean((pred_actions - actions_batch) ** 2, dim=-1)
            all_errors.append(errors.cpu())
            
            n_batches += 1
    
    # Aggregate metrics
    avg_loss = total_loss / n_batches
    all_errors = torch.cat(all_errors)
    
    results = {
        'mse_loss': avg_loss,
        'rmse': np.sqrt(avg_loss),
        'mae': torch.mean(torch.abs(torch.cat([
            torch.abs(p - t) for p, t in 
            zip(all_predictions if return_predictions else [], Y_test.split(batch_size))
        ]))).item() if return_predictions else None,
        'error_std': all_errors.std().item(),
        'error_max': all_errors.max().item(),
        'error_percentiles': {
            '50': all_errors.median().item(),
            '90': all_errors.quantile(0.9).item(),
            '95': all_errors.quantile(0.95).item(),
            '99': all_errors.quantile(0.99).item(),
        }
    }
    
    if return_predictions:
        results['predictions'] = torch.cat(all_predictions).numpy()
    
    return results


def fine_tune_bc(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    freeze_layers: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fine-tune a pre-trained model on new data.
    
    Args:
        model: Pre-trained model
        X: New states
        Y: New actions
        epochs: Number of fine-tuning epochs
        batch_size: Batch size
        lr: Learning rate (typically lower than initial training)
        freeze_layers: Optional list of layer names to freeze
        **kwargs: Additional arguments passed to fit_bc
        
    Returns:
        Training results dictionary
    """
    # Freeze specified layers
    if freeze_layers:
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
                logger.info(f"Froze layer: {name}")
    
    # Fine-tune with lower learning rate
    return fit_bc(
        model, X, Y, 
        epochs=epochs, 
        batch_size=batch_size, 
        lr=lr,
        **kwargs
    )


def train_ensemble(
    model_class: type,
    model_kwargs: Dict[str, Any],
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    n_models: int = 5,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    bootstrap: bool = True,
    **kwargs
) -> List[nn.Module]:
    """
    Train an ensemble of models for uncertainty estimation.
    
    Args:
        model_class: Class of model to instantiate
        model_kwargs: Arguments for model initialization
        X: Training states
        Y: Training actions
        n_models: Number of models in ensemble
        epochs: Training epochs per model
        batch_size: Batch size
        lr: Learning rate
        bootstrap: Whether to use bootstrap sampling
        **kwargs: Additional arguments passed to fit_bc
        
    Returns:
        List of trained models
    """
    ensemble = []
    n_samples = len(X)
    
    for i in range(n_models):
        logger.info(f"Training ensemble model {i+1}/{n_models}")
        
        # Create model
        model = model_class(**model_kwargs)
        
        # Bootstrap sampling if requested
        if bootstrap:
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices] if isinstance(X, np.ndarray) else X[indices]
            Y_boot = Y[indices] if isinstance(Y, np.ndarray) else Y[indices]
        else:
            X_boot, Y_boot = X, Y
        
        # Train model
        fit_bc(
            model, X_boot, Y_boot,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=False,
            **kwargs
        )
        
        ensemble.append(model)
    
    return ensemble


class SSEProgressCallback:
    """
    Server-Sent Events progress callback for streaming training updates.
    
    This callback streams training progress through SSE for real-time monitoring
    in web interfaces or other clients.
    """
    
    def __init__(
        self,
        queue: Optional[Queue] = None,
        include_gradients: bool = False,
        include_weights: bool = False,
        update_frequency: int = 1,
        buffer_size: int = 100
    ):
        """
        Initialize SSE progress callback.
        
        Args:
            queue: Queue to send events to (creates new if None)
            include_gradients: Whether to include gradient statistics
            include_weights: Whether to include weight statistics
            update_frequency: Send update every N batches
            buffer_size: Maximum queue size
        """
        self.queue = queue or Queue(maxsize=buffer_size)
        self.include_gradients = include_gradients
        self.include_weights = include_weights
        self.update_frequency = update_frequency
        self.batch_count = 0
        self.start_time = time.time()
        self.epoch_start_time = None
        
    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        batch_idx: Optional[int] = None,
        batch_loss: Optional[float] = None,
        **kwargs
    ):
        """Send training update via SSE."""
        event = self._create_event(
            model, epoch, train_loss, val_loss,
            batch_idx, batch_loss, **kwargs
        )
        
        # Send event to queue
        try:
            self.queue.put_nowait(event)
        except:
            # Queue full, drop oldest and retry
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(event)
            except:
                pass
    
    def _create_event(self, model, epoch, train_loss, val_loss, batch_idx, batch_loss, **kwargs):
        """Create SSE event data."""
        event = {
            'type': 'training_update',
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'train_loss': float(train_loss) if train_loss is not None else None,
            'val_loss': float(val_loss) if val_loss is not None else None,
            'elapsed_time': time.time() - self.start_time
        }
        
        # Add batch-level info if available
        if batch_idx is not None:
            event['batch_idx'] = batch_idx
            event['batch_loss'] = float(batch_loss) if batch_loss is not None else None
        
        # Add gradient statistics if requested
        if self.include_gradients and model is not None:
            grad_stats = self._compute_gradient_stats(model)
            if grad_stats:
                event['gradients'] = grad_stats
        
        # Add weight statistics if requested
        if self.include_weights and model is not None:
            weight_stats = self._compute_weight_stats(model)
            if weight_stats:
                event['weights'] = weight_stats
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in event:
                try:
                    event[key] = float(value) if isinstance(value, (int, float)) else str(value)
                except:
                    event[key] = str(value)
        
        return event
    
    def _compute_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient statistics."""
        total_norm = 0.0
        grad_norms = []
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        if grad_norms:
            return {
                'total_norm': total_norm,
                'mean_norm': np.mean(grad_norms),
                'max_norm': max(grad_norms),
                'min_norm': min(grad_norms)
            }
        return {}
    
    def _compute_weight_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute weight statistics."""
        weight_norms = []
        
        for param in model.parameters():
            if param.requires_grad:
                weight_norms.append(param.data.norm(2).item())
        
        if weight_norms:
            return {
                'mean_norm': np.mean(weight_norms),
                'max_norm': max(weight_norms),
                'min_norm': min(weight_norms),
                'std_norm': np.std(weight_norms)
            }
        return {}
    
    def get_event(self, timeout: Optional[float] = None):
        """Get next event from queue."""
        return self.queue.get(timeout=timeout)
    
    def format_sse(self, event: Dict) -> str:
        """Format event as SSE string."""
        data = json.dumps(event)
        return f"data: {data}\n\n"


@dataclass
class BatchProgressCallback:
    """
    Callback for batch-level progress updates.
    """
    callback: Callable
    update_frequency: int = 1
    _batch_count: int = 0
    
    def __call__(self, batch_idx: int, batch_loss: float, model: nn.Module = None, **kwargs):
        """Call callback with batch progress."""
        self._batch_count += 1
        if self._batch_count % self.update_frequency == 0:
            self.callback(
                model=model,
                batch_idx=batch_idx,
                batch_loss=batch_loss,
                **kwargs
            )


def create_sse_streamer(
    host: str = 'localhost',
    port: int = 5000,
    endpoint: str = '/training/stream'
) -> Tuple[SSEProgressCallback, threading.Thread]:
    """
    Create SSE streaming server for training progress.
    
    Args:
        host: Server host
        port: Server port
        endpoint: SSE endpoint path
        
    Returns:
        Tuple of (SSEProgressCallback, server_thread)
        
    Example:
        callback, server = create_sse_streamer()
        
        # Start server
        server.start()
        
        # Train with callback
        fit_bc(model, X, Y, callbacks=[callback])
        
        # Client can connect to http://localhost:5000/training/stream
    """
    from flask import Flask, Response
    import flask
    
    app = Flask(__name__)
    callback = SSEProgressCallback()
    
    @app.route(endpoint)
    def stream():
        def generate():
            while True:
                try:
                    event = callback.get_event(timeout=1.0)
                    yield callback.format_sse(event)
                except:
                    # Send heartbeat
                    yield ': heartbeat\n\n'
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )
    
    # Create server thread
    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False),
        daemon=True
    )
    
    return callback, server_thread


def create_websocket_callback(
    host: str = 'localhost',
    port: int = 8765
) -> 'WebSocketProgressCallback':
    """
    Create WebSocket callback for real-time training updates.
    
    Args:
        host: WebSocket server host
        port: WebSocket server port
        
    Returns:
        WebSocketProgressCallback instance
    """
    return WebSocketProgressCallback(host, port)


class WebSocketProgressCallback:
    """
    WebSocket-based progress callback for real-time updates.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.server_thread = None
        
    def start_server(self):
        """Start WebSocket server."""
        import asyncio
        import websockets
        
        async def handler(websocket, path):
            self.clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.clients.remove(websocket)
        
        async def main():
            async with websockets.serve(handler, self.host, self.port):
                await asyncio.Future()  # Run forever
        
        def run_server():
            asyncio.run(main())
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def __call__(self, model, epoch, train_loss, val_loss=None, **kwargs):
        """Send update to all connected clients."""
        import asyncio
        import websockets
        
        event = {
            'type': 'training_update',
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'train_loss': float(train_loss) if train_loss else None,
            'val_loss': float(val_loss) if val_loss else None,
            **kwargs
        }
        
        message = json.dumps(event)
        
        # Send to all clients
        if self.clients:
            asyncio.run(self._broadcast(message))
    
    async def _broadcast(self, message):
        """Broadcast message to all clients."""
        import websockets
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )


def create_tensorboard_callback(
    log_dir: str = './runs',
    experiment_name: Optional[str] = None
) -> 'TensorBoardCallback':
    """
    Create TensorBoard callback for training visualization.
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name for this experiment
        
    Returns:
        TensorBoardCallback instance
    """
    return TensorBoardCallback(log_dir, experiment_name)


class TensorBoardCallback:
    """
    TensorBoard logging callback.
    """
    
    def __init__(self, log_dir: str = './runs', experiment_name: Optional[str] = None):
        from torch.utils.tensorboard import SummaryWriter
        
        if experiment_name:
            log_dir = f"{log_dir}/{experiment_name}"
        
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
    
    def __call__(self, model, epoch, train_loss, val_loss=None, **kwargs):
        """Log metrics to TensorBoard."""
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        
        if val_loss is not None:
            self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Log learning rate if available
        if 'lr' in kwargs:
            self.writer.add_scalar('Learning_Rate', kwargs['lr'], epoch)
        
        # Log gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                self.writer.add_scalar(f'GradientNorms/{name}', param.grad.norm(), epoch)
        
        # Log weights
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Weights/{name}', param, epoch)
        
        self.global_step += 1
        self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


def fit_bc_with_streaming(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    stream_port: int = 5000,
    use_tensorboard: bool = False,
    **kwargs
) -> Tuple[Dict[str, Any], SSEProgressCallback]:
    """
    Fit model with SSE streaming of progress.
    
    This is a convenience wrapper around fit_bc that automatically sets up
    SSE streaming for real-time progress monitoring.
    
    Args:
        model: Neural network model
        X: Training states
        Y: Training actions
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        stream_port: Port for SSE streaming server
        use_tensorboard: Whether to also log to TensorBoard
        **kwargs: Additional arguments for fit_bc
        
    Returns:
        Tuple of (training_results, sse_callback)
        
    Example:
        results, streamer = fit_bc_with_streaming(
            model, X, Y,
            epochs=30,
            stream_port=5000
        )
        
        # Client can connect to http://localhost:5000/training/stream
        # to receive real-time updates
    """
    callbacks = kwargs.get('callbacks', [])
    
    # Create SSE callback and server
    sse_callback, server = create_sse_streamer(port=stream_port)
    callbacks.append(sse_callback)
    
    # Add TensorBoard if requested
    if use_tensorboard:
        tb_callback = create_tensorboard_callback()
        callbacks.append(tb_callback)
    
    # Start streaming server
    server.start()
    logger.info(f"SSE streaming server started on http://localhost:{stream_port}/training/stream")
    
    # Run training
    kwargs['callbacks'] = callbacks
    results = fit_bc(model, X, Y, epochs=epochs, batch_size=batch_size, lr=lr, **kwargs)
    
    # Send completion event
    sse_callback({
        'type': 'training_complete',
        'final_train_loss': results['final_train_loss'],
        'final_val_loss': results['final_val_loss'],
        'best_val_loss': results['best_val_loss'],
        'training_time': results['training_time']
    })
    
    return results, sse_callback


def compute_bc_metrics(
    model: nn.Module,
    env: Any,
    n_episodes: int = 10,
    max_steps: int = 1000,
    deterministic: bool = True,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute performance metrics by running the trained model in the environment.
    
    Args:
        model: Trained BC model
        env: Environment to evaluate in
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        device: Device for model
        
    Returns:
        Dictionary with performance metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    episode_returns = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_return = 0
        
        for t in range(max_steps):
            # Get action from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if hasattr(model, 'get_action'):
                    action = model.get_action(state_tensor, deterministic=deterministic)
                else:
                    action, _ = model(state_tensor, deterministic=deterministic)
                    action = action.squeeze(0)
            
            # Convert to numpy if needed
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            state = next_state
            
            if done:
                if info.get('success', False):
                    success_count += 1
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(t + 1)
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths
    }


# Example usage
if __name__ == "__main__":
    # Create dummy data
    n_samples = 1000
    state_dim = 10
    action_dim = 4
    
    X = np.random.randn(n_samples, state_dim).astype(np.float32)
    Y = np.random.randn(n_samples, action_dim).astype(np.float32)
    
    # Create model
    from cogniforge.core.policy import SimplePolicy
    model = SimplePolicy(obs_dim=state_dim, act_dim=action_dim, continuous=True)
    
    print("=" * 60)
    print("Testing fit_bc function")
    print("=" * 60)
    
    # Train model
    results = fit_bc(
        model, X, Y,
        epochs=20,
        batch_size=128,
        lr=1e-3,
        val_split=0.2,
        early_stopping=True,
        patience=5,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Final train loss: {results['final_train_loss']:.6f}")
    print(f"Final val loss: {results['final_val_loss']:.6f}")
    print(f"Best val loss: {results['best_val_loss']:.6f} at epoch {results['best_epoch']+1}")
    print(f"Training time: {results['training_time']:.1f} seconds")
    
    # Test evaluation
    X_test = np.random.randn(200, state_dim).astype(np.float32)
    Y_test = np.random.randn(200, action_dim).astype(np.float32)
    
    print("\n" + "=" * 60)
    print("Testing evaluate_bc function")
    print("=" * 60)
    
    eval_results = evaluate_bc(model, X_test, Y_test)
    print(f"Test MSE: {eval_results['mse_loss']:.6f}")
    print(f"Test RMSE: {eval_results['rmse']:.6f}")
    print(f"Error std: {eval_results['error_std']:.6f}")
    print(f"90th percentile error: {eval_results['error_percentiles']['90']:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)