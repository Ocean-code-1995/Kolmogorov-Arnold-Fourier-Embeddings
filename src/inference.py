import os
import json
import torch
import argparse
from models.vision_transformer import ViT  # Ensure this imports your model definition

def load_hyperparameters(hparams_path):
    """Load hyperparameters from a JSON file."""
    with open(hparams_path, "r") as fp:
        hyperparams = json.load(fp)
    return hyperparams

def load_model(checkpoint_path, hparams_path, device="cpu"):
    """
    Loads the model for inference.
    
    Args:
        checkpoint_path (str): Path to the saved model weights (.ckpt file).
        hparams_path (str): Path to the JSON file with hyperparameters.
        device (str): Device to load the model onto ("cpu" or "cuda").
    
    Returns:
        model: The loaded model ready for inference.
    """
    # Load hyperparameters used during training.
    hparams = load_hyperparameters(hparams_path)
    
    # Instantiate the model using the hyperparameters.
    model = ViT(
        img_size       = hparams["img_size"],
        patch_size     = hparams["patch_size"],
        in_channels    = hparams["in_channels"],
        embed_size     = hparams["embed_size"],
        depth          = hparams["depth"],
        heads          = hparams["heads"],
        mlp_dim        = hparams["mlp_dim"],
        dropout        = hparams["dropout"],
        learning_rate  = hparams["learning_rate"],
        embedding_type = hparams["embedding_type"],
        fourier_params = hparams["fourier_params"]
    )

    # Load the checkpoint (model weights)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode.
    model.eval()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a trained model for inference using its identifier.")

    # Instead of specifying full paths, require the model name and identifier.
    parser.add_argument(
        "--embedding_type", type=str, required=True,
        help="Type of embedding used by the model (e.g., 'mlp', 'cnn', 'fourier')."
    )
    parser.add_argument(
        "--id", type=str, required=True,
        help="Unique identifier (timestamp) for the model files (e.g., '20230510_153045')."
    )
    parser.add_argument(
        "--model_name", type=str, default="ViT",
        help="Name of the model (e.g., 'ViT', 'CNN')."
    )
    parser.add_argument(
        "--base_dir", type=str, default="experiments/checkpoints/ViT",
        help="Base directory where checkpoints are stored."
    )
    parser.add_argument(
        "--device", type=str, default="gpu",
        help="Device to load the model on ('cpu' or 'cuda')."
    )
    args = parser.parse_args()

    # Construct the full paths based on the provided identifier and model name.
    # Assuming files are stored in: base_dir/model_name/model_name_final_<id>.ckpt (and .json)
    checkpoint_path = os.path.join(
        args.base_dir, args.model_name, args.embedding_type, args.id, f"{args.model_name}.ckpt"
    )
    hparams_path = os.path.join(
        args.base_dir, args.model_name, args.embedding_type, args.id, "_hparams.json"
    )

    # Print the constructed paths for debugging.
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Loading hyperparameters from: {hparams_path}")

    # Load the model based on provided paths.
    model = load_model(checkpoint_path, hparams_path, device=args.device)
    print("✅ Model loaded and ready for inference.")

    # Example inference:
    # For instance, if your model expects an input of shape (1, 3, 300, 300), you might do:
    # sample_input = torch.randn(1, 3, 300, 300).to(args.device)
    # outputs = model(sample_input)
    # print(outputs)
