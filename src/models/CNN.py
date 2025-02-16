import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.optim import Adam

class CNNClassifier(pl.LightningModule):
    def __init__(self, img_size=300, in_channels=1, num_classes=1, learning_rate=1e-3, conv_channels=None, dropout=0.5):
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        super().__init__()
        self.learning_rate = learning_rate
        self.img_size = img_size

        # Build a simple CNN feature extractor
        # Three convolutional blocks with Conv2d, BatchNorm, ReLU and MaxPool
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample by factor of 2

            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Compute the size of the flattened features.
        # After 3 pool layers, the spatial dimensions are reduced by 2^3 = 8.
        final_size = img_size // 8
        self.flatten_dim = conv_channels[-1] * (final_size ** 2)

        # Classifier head (MLP)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)  # For binary classification, num_classes=1
        )

        # Metrics for logging
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.train_f1 = F1Score(task="binary", average='micro')
        self.val_f1 = F1Score(task="binary", average='micro')
        self.test_f1 = F1Score(task="binary", average='micro')

        self.train_precision = Precision(task="binary", average='micro')
        self.val_precision = Precision(task="binary", average='micro')
        self.test_precision = Precision(task="binary", average='micro')

        self.train_recall = Recall(task="binary", average='micro')
        self.val_recall = Recall(task="binary", average='micro')
        self.test_recall = Recall(task="binary", average='micro')

        # Loss function for binary classification with logits
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Lists to manually store metrics for later visualization
        self.train_losses = []
        self.val_losses   = []

        self.train_f1_scores = []
        self.val_f1_scores = []

        self.train_precisions = []
        self.val_precisions = []

        self.train_recalls = []
        self.val_recalls = []

    def forward(self, x):
        """
        Forward pass through the CNN.
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        # Squeeze the last dimension in case logits is of shape (B, 1)
        return logits.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)  # Convert logits to probabilities

        acc = self.train_accuracy(preds, y)
        f1 = self.train_f1(preds, y)
        precision = self.train_precision(preds, y)
        recall = self.train_recall(preds, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'loss': loss,
            'train_f1': f1,
            'train_precision': precision,
            'train_recall': recall
        }

    def on_train_epoch_end(self):
        # Store average training metrics for the epoch
        avg_train_loss = self.trainer.callback_metrics["train_loss"].item()
        self.train_losses.append(avg_train_loss)
        
        avg_train_f1 = self.trainer.callback_metrics["train_f1"].item()
        self.train_f1_scores.append(avg_train_f1)
        
        avg_train_precision = self.trainer.callback_metrics["train_precision"].item()
        self.train_precisions.append(avg_train_precision)
        
        avg_train_recall = self.trainer.callback_metrics["train_recall"].item()
        self.train_recalls.append(avg_train_recall)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)

        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1(preds, y)
        precision = self.val_precision(preds, y)
        recall = self.val_recall(preds, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall
        }

    def on_validation_epoch_end(self):
        # Store average validation metrics for the epoch
        avg_val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(avg_val_loss)
        
        avg_val_f1 = self.trainer.callback_metrics["val_f1"].item()
        self.val_f1_scores.append(avg_val_f1)
        
        avg_val_precision = self.trainer.callback_metrics["val_precision"].item()
        self.val_precisions.append(avg_val_precision)
        
        avg_val_recall = self.trainer.callback_metrics["val_recall"].item()
        self.val_recalls.append(avg_val_recall)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)

        acc = self.test_accuracy(preds, y)
        f1 = self.test_f1(preds, y)
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
