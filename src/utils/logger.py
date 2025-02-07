import json
import os
import datetime
import torch
import yaml
import gc
from typing import Optional

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
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M")

    # Create a directory for the model logs (if not already present)
    model_dir = os.path.join(log_dir, f"{model_name}_{date}")
    os.makedirs(model_dir, exist_ok=True)

    # Save hyperparameters to a JSON file
    hparams_file = os.path.join(model_dir, "hparams.json")
    with open(hparams_file, "w") as file:
        json.dump(hparams, file, indent=4)

    print(f"Hyperparameters logged locally to {hparams_file}")


def clear_memory(model=None):
    """
    Deletes the model (if provided), clears the PyTorch CUDA cache,
    and forces garbage collection.
    """
    if model is not None:
        del model
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared.")


def save_final_model_and_hparams(
    model,
    embedding_type,
    hyperparams,
    model_name,
    base_checkpoint_dir="experiments/checkpoints/ViT",
    auto_save: Optional[bool] = None,  # Can be True, False, or None
    auto_clear: Optional[bool] = None, # Can be True, False, or None
):
    """
    Prompts (or auto-decides) whether to save the final model and hyperparameters.
    If saving, it stores the model weights and hyperparameters with a unique timestamp.
    """
    # Decide whether to save the model
    if auto_save is None:
        while True:
            decision = input("Do you want to save the final model? (y/n): ").strip().lower()
            if decision in ['y', 'n']:
                break
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    else:
        decision = 'y' if auto_save else 'n'

    unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if decision == 'y':
        checkpoint_dir = os.path.join(base_checkpoint_dir, model_name, embedding_type, unique_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path   = os.path.join(checkpoint_dir, f"{model_name}.ckpt")
        hparams_path = os.path.join(checkpoint_dir, "_hparams.json")
        
        try:
            torch.save(model.state_dict(), model_path)
            print(f"✅ Final model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

        with open(hparams_path, "w") as fp:
            json.dump(hyperparams, fp, indent=4)
        print(f"✅ Final hyperparameters saved to {hparams_path}")
    else:
        unpromising_dir = os.path.join("experiments/checkpoints", "unpromising_configs", embedding_type, model_name)
        os.makedirs(unpromising_dir, exist_ok=True)
        unpromising_hparams_path = os.path.join(unpromising_dir, f"{model_name}_{unique_id}.json")
        with open(unpromising_hparams_path, "w") as fp:
            json.dump(hyperparams, fp, indent=4)
        print(f"❌ Model not saved. Unpromising configuration saved to {unpromising_hparams_path}")

    # Decide whether to clear memory
    if auto_clear is None:
        while True:
            decision = input("Do you want to clear the memory and delete models and objects? (y/n): ").strip().lower()
            if decision in ['y', 'n']:
                break
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    else:
        decision = 'y' if auto_clear else 'n'

    if decision == 'y':
        clear_memory(model)



def save_experiment_artifacts(
    config: dict,
    model,
    model_type: str = 'ViT',
    embedding_type: str = 'linear',
    data_module_config: Optional[dict] = None,
    trainer_config: Optional[dict] = None,
    base_dir: str = "../configs/experiments"
):
    """
    Saves experiment artifacts into a unique subdirectory under base_dir.
    
    Artifacts include:
      - The unified configuration (data, model, training settings) in YAML.
      - Model weights (state_dict) or a full checkpoint.
      - Optionally, data module and trainer configurations.
      
    Parameters:
      config (dict): Unified experiment configuration.
      model: Trained model instance.
      data_module_config (dict, optional): Configuration/specs for the data module.
      trainer_config (dict, optional): Training settings or trainer hyperparameters.
      base_dir (str): Root directory for experiments.
      
    Returns:
      experiment_dir (str): The directory where all files were saved.
    """
    # Create a unique identifier using the current date/time.
    unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define the experiment directory.
    experiment_dir = os.path.join(base_dir, model_type, embedding_type, unique_id) \
                        if model_type == 'ViT' \
                        else os.path.join(base_dir, model_type, unique_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save the unified configuration to a YAML file in corresponding directory for embedding type.
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")
    
    # Save model weights (state_dict).
    model_weights_path = os.path.join(experiment_dir, "model_weights.ckpt")
    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}")
    
    # Optionally save data module configuration.
    if data_module_config is not None:
        data_config_path = os.path.join(experiment_dir, "data_module_config.yaml")
        with open(data_config_path, "w") as f:
            yaml.dump(data_module_config, f, default_flow_style=False)
        print(f"Data module configuration saved to {data_config_path}")
    
    # Optionally save trainer configuration.
    if trainer_config is not None:
        trainer_config_path = os.path.join(experiment_dir, "trainer_config.yaml")
        with open(trainer_config_path, "w") as f:
            yaml.dump(trainer_config, f, default_flow_style=False)
        print(f"Trainer configuration saved to {trainer_config_path}")
    
    # Optionally clear memory after saving.
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"All experiment artifacts saved in {experiment_dir}")
    return experiment_dir