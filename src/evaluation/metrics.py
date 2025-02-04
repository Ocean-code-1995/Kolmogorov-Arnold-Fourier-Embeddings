from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }


def evaluate_model(model, data_module, device, threshold=0.5, split='test'):
    model.to(device)  # Ensure model is on the correct device
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    # Use the appropriate dataloader from the data module
    dataloader = data_module.test_dataloader() if split == 'test' \
                 else data_module.val_dataloader() if split == 'val' \
                 else data_module.train_dataloader()

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > threshold).long()  # Apply sigmoid and threshold

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return calculate_metrics(all_labels, all_preds)