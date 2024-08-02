import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

data = np.load("../ml_data/ttv_detection_data.npz")
X = data['X']
y = data['y']
ttvs = data['ttvs']

# Ensure all arrays have the same number of samples
min_samples = min(len(X), len(y), len(ttvs))
X = X[:min_samples]
y = y[:min_samples]
ttvs = ttvs[:min_samples]

# Normalize the input data
X = (X - np.mean(X)) / np.std(X)

# Normalize TTV values
ttvs = (ttvs - np.mean(ttvs)) / np.std(ttvs)

# Convert to PyTorch tensors
X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
y = torch.FloatTensor(y)
ttvs = torch.FloatTensor(ttvs)

# Split the data
train_split = int(0.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]
ttvs_train, ttvs_test = ttvs[:train_split], ttvs[train_split:]

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train, ttvs_train)
test_dataset = TensorDataset(X_test, y_test, ttvs_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

class GatedResidualNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedResidualNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding='same', dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels * 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x_a, x_b = torch.chunk(x, 2, dim=1)
        x = torch.tanh(x_a) * torch.sigmoid(x_b)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x + residual
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x.transpose(1, 2)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=4)
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        x = self.layer_norm(x)
        return x.transpose(1, 2)

class MemoryEfficientTCNModel(nn.Module):
    def __init__(self, input_shape, nb_filters, kernel_size, nb_stacks, dilations):
        super(MemoryEfficientTCNModel, self).__init__()
        self.conv_blocks = nn.ModuleList()
        self.nb_stacks = nb_stacks

        in_channels = input_shape[0]
        for _ in range(nb_stacks):
            for dilation in dilations:
                self.conv_blocks.append(GatedResidualNetwork(in_channels, nb_filters, kernel_size, dilation))
                in_channels = nb_filters

        self.attention = AttentionBlock(nb_filters)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(nb_filters * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.classification_output = nn.Linear(32, 1)
        self.regression_output = nn.Linear(32, 1)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = self.attention(x)

        x_gap = self.global_avg_pool(x).squeeze(-1)
        x_last = x[:, :, -1]
        x = torch.cat([x_gap, x_last], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        classification_output = self.classification_output(x).squeeze(-1)
        regression_output = self.regression_output(x).squeeze(-1)

        return classification_output, regression_output

# Create model
input_shape = (1, X.shape[2])
nb_filters = 48
kernel_size = 3
nb_stacks = 2
dilations = [1, 2, 4, 8, 16]

model = MemoryEfficientTCNModel(input_shape, nb_filters, kernel_size, nb_stacks, dilations)
model = nn.DataParallel(model)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)
scaler = GradScaler()

# Custom loss function
def custom_loss(y_pred, y_true, ttv_true):
    classification_pred, regression_pred = y_pred
    bce_loss = F.binary_cross_entropy_with_logits(classification_pred, y_true)
    positive_mask = y_true == 1
    mse_loss = F.mse_loss(regression_pred[positive_mask], ttv_true[positive_mask])
    return bce_loss + 0.0001 * mse_loss

# Function to save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# Function to load checkpoint
def load_checkpoint(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filename}")
    return epoch, loss

# Training loop
num_epochs = 50
accumulation_steps = 8
save_every = 5  # Save every 5 epochs

train_losses = []
val_losses = []


# Check if there's a checkpoint to resume from
latest_checkpoint = max([f for f in os.listdir('.') if f.startswith('checkpoint_')], default=None)
start_epoch = 0
if latest_checkpoint:
    start_epoch, _ = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
    start_epoch += 1  # Start from the next epoch

for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    for i, (batch_X, batch_y, batch_ttv) in enumerate(train_loader):
        batch_X, batch_y, batch_ttv = batch_X.to(device), batch_y.to(device), batch_ttv.to(device)

        with autocast():
            outputs = model(batch_X)
            loss = custom_loss(outputs, batch_y, batch_ttv) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps

    train_losses.append(epoch_loss / len(train_loader))

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y, batch_ttv in test_loader:
            batch_X, batch_y, batch_ttv = batch_X.to(device), batch_y.to(device), batch_ttv.to(device)
            with autocast():
                val_outputs = model(batch_X)
                val_loss += custom_loss(val_outputs, batch_y, batch_ttv).item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Save checkpoint every 'save_every' epochs
    if (epoch + 1) % save_every == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, f'checkpoint_epoch_{epoch+1}.pth')

# Save the final model
torch.save(model.state_dict(), 'optimized_ttv_detection_model.pth')
print("Final model saved as 'optimized_ttv_detection_model.pth'")


# Make predictions
model.eval()
with torch.no_grad():
    classification_pred, regression_pred = model(X_test.to(device))
    classification_pred = torch.sigmoid(classification_pred).cpu().numpy()  # Apply sigmoid here for probabilities
    regression_pred = regression_pred.cpu().numpy()

# Evaluate the model
y_pred = (classification_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")



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

# Add value labels on top of each bar
for i, v in enumerate(values):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# Save the plot
save_plot(fig, ax, 'evaluation_metrics.png')

print("Evaluation metrics plot has been saved as 'evaluation_metrics.png'.")

# Plot training history
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_losses, label='Training Loss')
ax.plot(val_losses, label='Validation Loss')
ax.set_title('Model Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
save_plot(fig, ax, 'model_loss.png')

# Plot predicted vs true TTVs
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ttvs_test[y_test == 1], regression_pred[y_test == 1], alpha=0.5)
ax.plot([ttvs_test.min(), ttvs_test.max()], [ttvs_test.min(), ttvs_test.max()], 'r--', lw=2)
ax.set_title('Predicted vs True TTVs')
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
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, classification_pred)
roc_auc = auc(fpr, tpr)
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