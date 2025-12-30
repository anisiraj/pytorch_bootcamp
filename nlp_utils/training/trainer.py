"""Training loop implementation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics


class Trainer:
    """
    Simple trainer for PyTorch models with torchmetrics support.

    Handles training loop, validation, metrics computation, and history tracking.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer (e.g., Adam, SGD)
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
        device: Device to train on (cuda/cpu)
        metrics: Dict of torchmetrics.Metric objects {'name': metric}
                 If None, defaults to accuracy for classification
        grad_clip_norm: Maximum gradient norm for clipping (default: 1.0)

    Example:
        >>> from nlp_utils import SentimentModel, Trainer
        >>> import torchmetrics
        >>> model = SentimentModel(vocab)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> metrics = {
        ...     'acc': torchmetrics.Accuracy(task='multiclass', num_classes=2),
        ...     'f1': torchmetrics.F1Score(task='multiclass', num_classes=2)
        ... }
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     device=device,
        ...     metrics=metrics
        ... )
        >>> history = trainer.fit(num_epochs=5)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        grad_clip_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.history = {}
        self.best_val_loss = float('inf')

        # Default to accuracy if no metrics provided
        if metrics is None:
            # Infer num_classes from model if possible
            num_classes = getattr(model, 'num_categories', 2)
            metrics = {
                'acc': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
            }

        # Move metrics to device
        self.metrics = {name: metric.to(device) for name, metric in metrics.items()}

    def _run_epoch(self, dataloader: DataLoader, is_training: bool = True) -> dict:
        """
        Run one epoch of training or validation.

        Args:
            dataloader: DataLoader to iterate over
            is_training: If True, update weights; if False, just evaluate

        Returns:
            Dict with 'loss' and all metric values
        """
        # Set model mode
        self.model.train() if is_training else self.model.eval()
        desc = "Training" if is_training else "Validation"

        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()

        # Accumulators
        total_loss = 0

        # Context manager for gradients
        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            for batch in tqdm(dataloader, desc=desc):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Zero gradients
                if is_training:
                    self.optimizer.zero_grad()

                # Forward
                logits = self.model(batch)
                loss = self.loss_fn(logits, batch['label'])

                # Backward
                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm
                    )
                    self.optimizer.step()

                # Update metrics
                total_loss += loss.item()
                predictions = logits.argmax(dim=1)
                for metric in self.metrics.values():
                    metric.update(predictions, batch['label'])

        # Compute results
        results = {'loss': total_loss / len(dataloader)}

        # Compute metrics
        for name, metric in self.metrics.items():
            results[name] = metric.compute().item()

        return results

    def fit(self, num_epochs: int = 5) -> dict:
        """
        Train the model for multiple epochs with validation.

        Validates after every epoch and tracks best validation loss.

        Args:
            num_epochs: Number of training epochs

        Returns:
            Dictionary with training history containing:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'train_{metric}': Training metric values per epoch
            - 'val_{metric}': Validation metric values per epoch
        """
        # Initialize history
        self.history = {'train_loss': [], 'val_loss': []}
        for metric_name in self.metrics.keys():
            self.history[f'train_{metric_name}'] = []
            self.history[f'val_{metric_name}'] = []

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_results = self._run_epoch(self.train_loader, is_training=True)

            # Validate
            val_results = self._run_epoch(self.val_loader, is_training=False)

            # Update history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            for metric_name in self.metrics.keys():
                self.history[f'train_{metric_name}'].append(train_results[metric_name])
                self.history[f'val_{metric_name}'].append(val_results[metric_name])

            # Print results
            self._print_epoch_results(train_results, val_results)

            # Track best model
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                print(f"  ✓ New best validation loss!")

        return self.history

    def _print_epoch_results(self, train_results: dict, val_results: dict):
        """Pretty print epoch results."""
        print(f"\nResults:")

        # Train metrics
        print(f"  Train - Loss: {train_results['loss']:.4f}", end="")
        for name in self.metrics.keys():
            print(f" | {name}: {train_results[name]:.4f}", end="")
        print()

        # Val metrics
        print(f"  Val   - Loss: {val_results['loss']:.4f}", end="")
        for name in self.metrics.keys():
            print(f" | {name}: {val_results[name]:.4f}", end="")
        print()