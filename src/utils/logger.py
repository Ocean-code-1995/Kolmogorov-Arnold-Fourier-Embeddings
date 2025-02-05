import json
import os
import datetime
import torch

def log_hyperparameters(hparams, model_name, log_dir="experiments/configs"):
    """
    Store hyperparameters in a JSON file with model name and date as an indicator
    in the filename, stored in the specified directory (log_dir).

    Parameters:
        hparams (dict): Dictionary of hyperparameters.
        model_name (str): Name of the model (e.g., 'ViT', 'CNN').
        log_dir (str): Directory where the logs will be stored.
    """
    # Create a timestamp
    now = datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M")

    # Create a directory for the model logs (if not already present)
    model_dir = os.path.join(log_dir, f"{model_name}_{date}")
    os.makedirs(model_dir, exist_ok=True)

    # Save hyperparameters to a JSON file
    hparams_file = os.path.join(model_dir, "hparams.json")
    with open(hparams_file, "w") as file:
        json.dump(hparams, file, indent=4)

    print(f"Hyperparameters logged locally to {hparams_file}")


def save_final_model_and_hparams(
    model,
    embedding_type,
    hyperparams,
    model_name,
    base_checkpoint_dir="experiments/checkpoints/ViT"
    ):
    """
    Prompts the user to decide whether to save the final model and hyperparameters.
    If the user inputs 'y', saves the model weights and hyperparameters with a unique timestamp.

    Args:
        model: The trained model.
        hyperparams: Dictionary containing hyperparameters.
        model_name: Name of the model (e.g., "ViT" or "CNN").
        base_checkpoint_dir: Base directory for storing checkpoints.
    """
    # Loop until the user provides a valid input.
    while True:
        decision = input(
            "Do you want to save the final model based on these loss curves? (y/n): "
        ).strip().lower()
        if decision in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")


    # Create a unique identifier using the current date and time.
    unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if decision == 'y':

        # Define a checkpoint directory specific to the model.
        checkpoint_dir = os.path.join(base_checkpoint_dir, model_name, embedding_type, unique_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Construct file paths with the unique identifier.
        model_path   = os.path.join(checkpoint_dir, f"{model_name}.ckpt")
        hparams_path = os.path.join(checkpoint_dir, "_hparams.json")

        # Save the model weights.
        torch.save(model.state_dict(), model_path)
        # Save the hyperparameters as a JSON file.
        with open(hparams_path, "w") as fp:
            json.dump(hyperparams, fp, indent=4)

        print(f"✅ Final model saved to {model_path}")
        print(f"✅ Final hyperparameters saved to {hparams_path}")

    else:
        # Save the hyperparameters to a dedicated "unpromising" folder.
        unpromising_dir = os.path.join(
            "experiments/checkpoints", "unpromising_configs", embedding_type, model_name
        )
        os.makedirs(unpromising_dir, exist_ok=True)

        unpromising_hparams_path = os.path.join(unpromising_dir, f"{model_name}_{unique_id}.json")

        with open(unpromising_hparams_path, "w") as fp:
            json.dump(hyperparams, fp, indent=4)

        print(f"❌ Model not saved. Unpromising configuration saved to {unpromising_hparams_path}")