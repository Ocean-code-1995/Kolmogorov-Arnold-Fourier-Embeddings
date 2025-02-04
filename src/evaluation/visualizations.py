import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import pandas as pd
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2



def plot_loss_curves(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    
    Parameters:
        train_losses (list or array): Loss values for each epoch during training.
        val_losses (list or array): Loss values for each epoch during validation.
    """
    # Create an epoch index based on the length of train losses.
    epochs = range(1, len(train_losses) + 1)
    
    # Calculate the difference between the final training and validation loss.
    delta_loss = train_losses[-1] - val_losses[-1]
    
    # Plotting properties
    line_width = 2
    train_color = "#007BFF"  # Blue for training
    val_color = "#FF4500"    # Red for validation
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color=train_color, linewidth=line_width)
    plt.plot(epochs, val_losses, label='Validation Loss', color=val_color, linewidth=line_width)
    plt.title(f'Loss Curves\nΔ = {delta_loss:.2f}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_metrics_curves(model):
    train_epochs = range(len(model.train_losses))
    val_epochs = range(len(model.val_losses))
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns

    # Common plot properties
    line_width = 4
    train_color = "#007BFF"
    val_color = "#FF4500"
    tick_label_size = 16
    legend_properties = {
        'facecolor': 'white',
        'edgecolor': 'black',
        'shadow': True,
        'loc': 'upper right',
        'fontsize': 18
    }

    # Calculate final deltas for each metric
    delta_loss = model.train_losses[-1] - model.val_losses[-1]
    delta_f1 = model.train_f1_scores[-1] - model.val_f1_scores[-1]
    delta_precision = model.train_precisions[-1] - model.val_precisions[-1]
    delta_recall = model.train_recalls[-1] - model.val_recalls[-1]

    # Plotting the loss curves
    axs[0].plot(train_epochs, model.train_losses, label='Training', color=train_color, linewidth=line_width)
    axs[0].plot(val_epochs, model.val_losses, label='Validation', color=val_color, linewidth=line_width)
    axs[0].set_title(f'Loss\nΔ={delta_loss:.2f}', fontsize=25, weight='bold', pad=15)

    axs[0].set_xlabel('Epochs', fontsize=20, weight='bold')
    axs[0].set_ylabel('Metric Value', fontsize=20, weight='bold')
    axs[0].tick_params(axis='both', which='major', labelsize=tick_label_size)
    axs[0].legend(**legend_properties)

    # Plotting the F1 score curves
    axs[1].plot(train_epochs, model.train_f1_scores, label='Training', color=train_color, linewidth=line_width)
    axs[1].plot(val_epochs, model.val_f1_scores, label='Validation', color=val_color, linewidth=line_width)
    axs[1].set_title(f'F1 Score\nΔ={delta_f1:.2f}', fontsize=25, weight='bold', pad=15)
    axs[1].set_xlabel('Epochs', fontsize=20, weight='bold')
    axs[1].set_ylim(0, 1)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_label_size)

    # Plotting the Precision curves
    axs[2].plot(train_epochs, model.train_precisions, label='Training', color=train_color, linewidth=line_width)
    axs[2].plot(val_epochs, model.val_precisions, label='Validation', color=val_color, linewidth=line_width)
    axs[2].set_title(f'Precision\nΔ={delta_precision:.2f}', fontsize=25, weight='bold', pad=15)
    axs[2].set_xlabel('Epochs', fontsize=20, weight='bold')
    axs[2].set_ylim(0, 1)
    axs[2].tick_params(axis='both', which='major', labelsize=tick_label_size)

    # Plotting the Recall curves
    axs[3].plot(train_epochs, model.train_recalls, label='Training', color=train_color, linewidth=line_width)
    axs[3].plot(val_epochs, model.val_recalls, label='Validation', color=val_color, linewidth=line_width)
    axs[3].set_title(f'Recall\nΔ={delta_recall:.2f}', fontsize=25, weight='bold', pad=15)
    axs[3].set_xlabel('Epochs', fontsize=20, weight='bold')
    axs[3].set_ylim(0, 1)
    axs[3].tick_params(axis='both', which='major', labelsize=tick_label_size)

    plt.tight_layout()
    plt.show();



def visualize_average_attention_maps(model, input_images):
    """
    Visualize the average attention maps from the Vision Transformer overlaid on the original images in a single row
    with a single colorbar at the end.

    Parameters:
    - model: The trained Vision Transformer model.
    - input_images: A list of four input image tensors (C, H, W).

    Returns:
    - None (displays the plot)
    """
    model.eval()
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))  # Create a row of 4 subplots

    for idx, input_image in enumerate(input_images):
        patch_size = model.patch_embedding.patch_size  # Dynamically get the patch size from the model
        img_size = input_image.shape[1]  # Assuming input_image is (C, H, W) and H == W

        with torch.no_grad():
            prediction, attn_maps = model(input_image.unsqueeze(0), return_attn=True)

        predicted_prob = torch.sigmoid(prediction).item()
        predicted_label = 'Healthy' if predicted_prob >= 0.5 else 'Bleached'

        # Use the last layer's attention map
        last_layer_attn = attn_maps[-1].squeeze(0)  # Assume batch size of 1 for simplicity

        # Calculate average attention map across all heads
        avg_attn_map = last_layer_attn.mean(0).cpu().numpy()

        # Reshape the attention map to match the number of patches
        num_patches = img_size // patch_size
        reshaped_attn_map = avg_attn_map[1:].reshape(num_patches, num_patches)  # Exclude the CLS token

        # Resize the attention map to match the image size using interpolation
        resized_attn_map = cv2.resize(reshaped_attn_map, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Normalize the resized attention map for better visualization
        resized_attn_map = (resized_attn_map - resized_attn_map.min()) / (resized_attn_map.max() - resized_attn_map.min())

        # Plot the original image and the attention map overlay
        original_img = input_image.permute(1, 2, 0).cpu().numpy()
        im = axes[idx].imshow(original_img)
        overlay = axes[idx].imshow(resized_attn_map, cmap='coolwarm', alpha=0.66)  # Adjust alpha to taste
        axes[idx].set_title(f"Pred: {predicted_label} (P: {predicted_prob:.2f})", fontsize=30, weight='bold')
        axes[idx].axis('off')

        # Only create the colorbar on the last subplot
        if idx == 3:
            divider = make_axes_locatable(axes[idx])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(overlay, cax=cax)
            cbar.ax.tick_params(labelsize=25)  # Set the size of the tick labels

    plt.tight_layout()
    plt.show();



def visualize_model_predictions(model, data_module):
    """
    Visualizes the prediction results of a pre-trained model by plotting the probability distributions
    for different quadrants of the confusion matrix using the test data provided by the data module.
    
    Parameters:
    - model: A pre-trained PyTorch model.
    - data_module: A PyTorch Lightning data module with a defined test dataloader.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    test_loader = data_module.test_dataloader()
    predictions, labels = [], []

    # Collect predictions and labels from the test dataset
    with torch.no_grad():
        for batch in test_loader:
            inputs, y = batch
            inputs = inputs.to(model.device)
            logits = model(inputs)
            y_prob = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid if the model outputs logits
            predictions.extend(y_prob)
            labels.extend(y.cpu().numpy())

    # Create a DataFrame for the predictions and actual labels
    df_predictions = pd.DataFrame({
        'y': labels,
        'y_proba': np.array(predictions).flatten()
    })

    # Setup plot dimensions and style
    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(8, 7))
    bins = np.linspace(0, 1, 21)

    # Helper function to plot each quadrant of the confusion matrix
    def show_quarter(df, query, col, title, ax, bins, color, alpha, x_label=None, y_label=None):
        results = df.query(query)
        sns.histplot(results[col], kde=False, ax=ax, bins=bins, color=color, alpha=alpha)
        if y_label:
            ax.set_ylabel(y_label, fontsize=18.5)
        if x_label:
            ax.set_xlabel(x_label, fontsize=18.5)
        ax.set_title(f'{title} ({results.shape[0]})', fontsize=18.5, weight='bold')
        ax.axvline(x=0.5, color='red', linestyle='-.', linewidth=2)

        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=15)

    # Visualize probabilities for each quadrant
    blue = "#007BFF"
    red = "#FF4500"
    show_quarter(df_predictions, "y==0 and y_proba < 0.5", "y_proba", "True Negatives", axs[0, 0], bins, blue, 0.6, y_label="False")
    show_quarter(df_predictions, "y==0 and y_proba >= 0.5", "y_proba", "False Positives", axs[0, 1], bins, red, 0.6)
    show_quarter(df_predictions, "y==1 and y_proba >= 0.5", "y_proba", "True Positives", axs[1, 1], bins, red, 0.6, x_label="Probability")
    show_quarter(df_predictions, "y==1 and y_proba < 0.5", "y_proba", "False Negatives", axs[1, 0], bins, blue, 0.6, x_label="Probability", y_label="True")

    # Configure common labels and title for the figure
    fig.text(0.5, 0.003, 'Predicted', ha='center', va='center', fontsize=18.5, weight='bold')
    fig.text(0.003, 0.5, 'True Label', ha='center', va='center', rotation='vertical', fontsize=18.5, weight='bold')
    #fig.suptitle("Model Predictions: Probabilities per Confusion Matrix Quadrant", fontsize=18)

    plt.tight_layout()
    plt.show()