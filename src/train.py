import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from models.vision_transformer import ViT
from data.data_module import CoralDataModule
from utils.logger import log_hyperparameters, save_final_model_and_hparams
from evaluation.visualizations import plot_loss_curves




@hydra_main(config_path="../configs", config_name="ViT_config")
def main(cfg: DictConfig) -> None:
    # Optional: print the configuration for debugging.
    print("==== Loaded Configuration ====")
    print(OmegaConf.to_yaml(cfg))

    # Hyperparameters to log
    hyperparams = {**cfg.data, **cfg.model, **cfg.training}


    # === Setup the DataModule ===
    data_module = CoralDataModule(
        data_dir    = cfg.data.data_dir,
        batch_size  = cfg.data.batch_size,
        scaler_type = cfg.data.scaler_type,
        img_size    = cfg.data.img_size,
        test_size   = cfg.data.test_size,
        val_size    = cfg.data.val_size,
        num_workers = cfg.data.num_workers,
        padding_mode= cfg.data.padding_mode
    )
    data_module.setup() # This will load the data and prepare it for training

    # Determine logging frequency
    num_batches = len(data_module.train_dataloader())
    log_every_n_steps = max(1, num_batches // 10)


    # === Initialize the Model ===
    model = ViT(
        img_size       = cfg.data.img_size,
        patch_size     = cfg.model.patch_size,
        in_channels    = cfg.model.in_channels,
        embed_size     = cfg.model.embed_size,
        depth          = cfg.model.depth,
        heads          = cfg.model.heads,
        mlp_dim        = cfg.model.mlp_dim,
        dropout        = cfg.model.dropout,
        learning_rate  = cfg.model.learning_rate,
        embedding_type = cfg.model.embedding_type,
        fourier_params = cfg.model.fourier_params
    )

    # === Setup WandB Logger ===
    wandb_logger = WandbLogger(
        project   = cfg.wandb.project,
        log_model = True
    )
    wandb_logger.log_hyperparams(hyperparams)

    # === Setup Checkpointing ===
    checkpoint_dir = os.path.join(
        "experiments", "checkpoints", cfg.model.model_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor    = 'val_loss',  # This is still useful to have
        dirpath    = checkpoint_dir,
        filename    = f"{cfg.model.model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k = 1, # =1 to save only the best model
        mode       = 'min'
    )

    # === Setup the Trainer ===
    trainer = pl.Trainer(
        max_epochs        = cfg.training.max_epochs,
        precision         = "16-mixed",
        devices           = 1 if torch.cuda.is_available() else 1,
        accelerator       = 'gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps = log_every_n_steps,
        logger            = wandb_logger,
        callbacks         = [checkpoint_callback]
    )

    # === Start Training ===
    trainer.fit(model, datamodule=data_module)

    # === Evaluate Loss on Train and Validation Sets ===

    # Plot the loss curves
    plot_loss_curves(
        train_losses = model.train_losses,
        val_losses   = model.val_losses,
    )

    # === Save the Final Model ===

    # Prompt User and Save Final Model and Hyperparameters if Desired
    save_final_model_and_hparams(
        model               = model,
        embedding_type      = cfg.model.embedding_type,
        hyperparams         = hyperparams,
        model_name          = cfg.model.model_name,
        base_checkpoint_dir = "experiments/checkpoints/ViT"
    )



if __name__ == "__main__":
    main()


# NOTE:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# when running the script xopu can override the parameters in the config file by passing them as arguments:

# e.g python src/train.py wandb.project="MyNewProject"
# or python src/train.py data.batch_size=64 model.depth=12
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~