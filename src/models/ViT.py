import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score
)
from torch.optim import Adam

from models.embeddings import PatchEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attn=False):
        x = self.norm1(x)
        x = x.permute(1, 0, 2)
        attn_output, attn_weights = self.attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)
        x = self.norm2(x.permute(1, 0, 2) + attn_output)
        x2 = self.mlp(x)
        return (x + x2, attn_weights) if return_attn else x + x2


class ViT(pl.LightningModule):
    def __init__(
        self, 
        img_size=300, 
        patch_size=4,
        in_channels=1,
        embed_size=64,
        depth=8,
        heads=4,
        mlp_dim=128,
        dropout=0.1, 
        learning_rate=1e-3,  
        embedding_type='conv',
        fourier_params=None
    ):
        super().__init__()
        
        # parameters
        self.learning_rate = learning_rate
        
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, in_channels, embed_size, embedding_type, fourier_params=fourier_params
        )
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_size, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, 1)  # Change to 1 for binary classification
        )
        
        # Metrics
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
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # manually store losses for plt/sns vizualization
        self.train_losses = []
        self.val_losses   = []
        
        self.train_f1_scores = []
        self.val_f1_scores = []
        
        self.train_precisions = []
        self.val_precisions = []
        
        self.train_recalls = []
        self.val_recalls = []
    
    def forward(self, x, return_attn=False):
        x = self.patch_embedding(x)
        attn_maps = []
        for block in self.transformer_blocks:
            if return_attn:
                x, attn_weights = block(x, return_attn=True)
                attn_maps.append(attn_weights)
            else:
                x = block(x)
        x = self.to_cls_token(x[:, 0])
        output = self.mlp_head(x).squeeze(-1)
        return (output, attn_maps) if return_attn else output
    
    def get_attention_maps(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if it's missing
        x = self.patch_embedding(x)
        attn_maps = []
        for i, block in enumerate(self.transformer_blocks):
            x, attn_weights = block(x, return_attn=True)
            print(f"Layer {i+1} attention shape: {attn_weights.shape}")
            attn_maps.append(attn_weights)
        return attn_maps

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
        # Store the average training loss for this epoch
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
        preds = torch.sigmoid(logits)  # Convert logits to probabilities
        
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
        # Store the average validation loss for this epoch
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
        preds = torch.sigmoid(logits)  # Convert logits to probabilities
        
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