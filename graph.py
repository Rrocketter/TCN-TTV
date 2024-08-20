import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have these defined/loaded:
# model = YourTrainedModel()
# X_test, y_test, ttvs_test = ...

# Convert data to PyTorch tensors if they aren't already
X_test = X_test.clone().detach() if isinstance(X_test, torch.Tensor) else torch.tensor(X_test, dtype=torch.float32)
y_test = y_test.clone().detach() if isinstance(y_test, torch.Tensor) else torch.tensor(y_test, dtype=torch.float32)
ttvs_test = ttvs_test.clone().detach() if isinstance(ttvs_test, torch.Tensor) else torch.tensor(ttvs_test,
                                                                                                dtype=torch.float32)

# Ensure all tensors are 1D
X_test = X_test.squeeze()
y_test = y_test.squeeze()
ttvs_test = ttvs_test.squeeze()

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the appropriate device
model = model.to(device)

# Set batch size
BATCH_SIZE = 32

# Evaluation
model.eval()
with torch.no_grad():
    classification_preds = []
    regression_preds = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch = X_test[i:i + BATCH_SIZE].to(device)
        classification_pred, regression_pred = model(batch)
        classification_preds.append(classification_pred.cpu())
        regression_preds.append(regression_pred.cpu())

    # Combine predictions
    classification_pred = torch.cat(classification_preds).squeeze()
    regression_pred = torch.cat(regression_preds).squeeze()

    classification_pred = torch.sigmoid(classification_pred).numpy()
    regression_pred = regression_pred.numpy()

# Convert to binary predictions
y_pred = (classification_pred > 0.5).astype(int)

# Ensure y_test is also a numpy array
y_test = y_test.numpy()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Plotting functions
def save_plot(fig, ax, filename):
    ax.figure.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Visualize evaluation metrics
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
ax.bar(metrics, values)
ax.set_ylim(0, 1)
ax.set_title('Model Evaluation Metrics')
ax.set_ylabel('Score')
for i, v in enumerate(values):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
save_plot(fig, ax, 'evaluation_metrics.png')
print("Evaluation metrics plot has been saved as 'evaluation_metrics.png'.")

# Plot predicted vs true TTVs
true_positive_mask = (y_test == 1) & (y_pred == 1)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ttvs_test[true_positive_mask].numpy(), regression_pred[true_positive_mask], alpha=0.5)
ax.plot([ttvs_test.min().item(), ttvs_test.max().item()], [ttvs_test.min().item(), ttvs_test.max().item()], 'r--', lw=2)
ax.set_title('Predicted vs True TTVs (True Positives)')
ax.set_xlabel('True TTV (minutes)')
ax.set_ylabel('Predicted TTV (minutes)')
save_plot(fig, ax, 'predicted_vs_true_ttvs.png')

# Plot TTV distribution
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(ttvs_test[y_test == 1].numpy(), bins=30, alpha=0.5, label='True TTVs')
ax.hist(regression_pred[y_pred == 1], bins=30, alpha=0.5, label='Predicted TTVs')
ax.set_title('TTV Distribution')
ax.set_xlabel('TTV (minutes)')
ax.set_ylabel('Frequency')
ax.legend()
save_plot(fig, ax, 'ttv_distribution.png')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, classification_pred)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
save_plot(fig, ax, 'roc_curve.png')

print("All plots have been saved as PNG files.")