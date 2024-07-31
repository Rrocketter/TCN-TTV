import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


# GPU Memory monitoring function
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# Load the preprocessed data
data = np.load("/kaggle/input/ttv-detection-data/ttv_detection_data.npz")
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
train_dataset = TensorDataset(X_train, y_train, ttvs_train)
test_dataset = TensorDataset(X_test, y_test, ttvs_test)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

# Model parameters
input_shape = (1, X.shape[2])  # (channels, time_steps)
nb_filters = 32
kernel_size = 3
nb_stacks = 2
dilations = [1, 2, 4, 8, 16]
output_len = 2  # Binary classification (TTV or no TTV) and TTV magnitude


def custom_loss(y_pred, y_true, ttv_true):
    classification_pred, regression_pred = y_pred

    # Binary cross-entropy loss with logits for classification
    bce_loss = F.binary_cross_entropy_with_logits(classification_pred, y_true)

    # Mean squared error loss for regression (only for positive TTV samples)
    mse_loss = F.mse_loss(regression_pred[y_true == 1], ttv_true[y_true == 1])

    # Combine losses with appropriate scaling
    total_loss = bce_loss + 0.01 * mse_loss  # Adjust the scaling factor as needed
    return total_loss


class GatedResidualNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedResidualNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels * 2, kernel_size,
                               padding='same', dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x_a, x_b = torch.chunk(x, 2, dim=1)
        x = torch.tanh(x_a) * torch.sigmoid(x_b)
        x = self.conv2(x)
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
    def __init__(self, input_shape, nb_filters, kernel_size, nb_stacks, dilations, output_len):
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
        self.classification_output = nn.Linear(32, 1)  # Output logits instead of probabilities
        self.regression_output = nn.Linear(32, 1)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = self.attention(x)

        x_gap = self.global_avg_pool(x).squeeze(-1)
        x_last = x[:, :, -1]
        x = torch.cat([x_gap, x_last], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        classification_output = self.classification_output(x).squeeze(-1)  # Output logits
        regression_output = self.regression_output(x).squeeze(-1)

        return classification_output, regression_output


# Create the model
model = MemoryEfficientTCNModel(input_shape, nb_filters, kernel_size, nb_stacks, dilations, output_len)
model = nn.DataParallel(model)

# Define optimizer and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Initialize the GradScaler for mixed precision training
scaler = GradScaler()

# Training loop
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y, batch_ttv in train_loader:
        batch_X, batch_y, batch_ttv = batch_X.to(device), batch_y.to(device), batch_ttv.to(device)

        with autocast():
            outputs = model(batch_X)
            loss = custom_loss(outputs, batch_y, batch_ttv)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Validation
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

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print_gpu_memory()

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

# Visualize results
plt.figure(figsize=(12, 8))

# Plot training history
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot predicted vs true TTVs
plt.subplot(2, 2, 2)
plt.scatter(ttvs_test[y_test == 1], regression_pred[y_test == 1], alpha=0.5)
plt.plot([ttvs_test.min(), ttvs_test.max()], [ttvs_test.min(), ttvs_test.max()], 'r--', lw=2)
plt.title('Predicted vs True TTVs')
plt.xlabel('True TTV (minutes)')
plt.ylabel('Predicted TTV (minutes)')

# Plot TTV distribution
plt.subplot(2, 2, 3)
plt.hist(ttvs_test[y_test == 1].numpy(), bins=30, alpha=0.5, label='True TTVs')
plt.hist(regression_pred[y_pred == 1], bins=30, alpha=0.5, label='Predicted TTVs')
plt.title('TTV Distribution')
plt.xlabel('TTV (minutes)')
plt.ylabel('Frequency')
plt.legend()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, classification_pred)
roc_auc = auc(fpr, tpr)

plt.subplot(2, 2, 4)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), 'ttv_detection_model.pth')
print("Model saved as 'ttv_detection_model.pth'")


# Kaggle Outout:
#
# Using device: cuda
# Epoch 1/50, Train Loss: 0.1899, Val Loss: 0.1425
# Learning rate: 0.000100
# GPU memory allocated: 0.02 GB
# GPU memory cached: 1.88 GB
# Epoch 2/50, Train Loss: 0.1792, Val Loss: 0.1396
# Learning rate: 0.000100
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 3/50, Train Loss: 0.1764, Val Loss: 0.1417
# Learning rate: 0.000100
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 4/50, Train Loss: 0.1725, Val Loss: 0.1410
# Learning rate: 0.000100
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 5/50, Train Loss: 0.1697, Val Loss: 0.1417
# Learning rate: 0.000100
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 6/50, Train Loss: 0.1684, Val Loss: 0.1411
# Learning rate: 0.000050
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 7/50, Train Loss: 0.1670, Val Loss: 0.1452
# Learning rate: 0.000050
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 8/50, Train Loss: 0.1666, Val Loss: 0.1414
# Learning rate: 0.000050
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 9/50, Train Loss: 0.1663, Val Loss: 0.1405
# Learning rate: 0.000050
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 10/50, Train Loss: 0.1661, Val Loss: 0.1416
# Learning rate: 0.000025
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 11/50, Train Loss: 0.1655, Val Loss: 0.1411
# Learning rate: 0.000025
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Epoch 12/50, Train Loss: 0.1653, Val Loss: 0.1411
# Learning rate: 0.000025
# GPU memory allocated: 0.03 GB
# GPU memory cached: 1.88 GB
# Early stopping triggered after 12 epochs
# ---------------------------------------------------------------------------
# OutOfMemoryError                          Traceback (most recent call last)
# Cell In[1], line 249
#     247 model.eval()
#     248 with torch.no_grad():
# --> 249     classification_pred, regression_pred = model(X_test.to(device))
#     250     classification_pred = torch.sigmoid(classification_pred).cpu().numpy()  # Apply sigmoid here for probabilities
#     251     regression_pred = regression_pred.cpu().numpy()
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
#    1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
#    1517 else:
# -> 1518     return self._call_impl(*args, **kwargs)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1527, in Module._call_impl(self, *args, **kwargs)
#    1522 # If we don't have any hooks, we want to skip the rest of the logic in
#    1523 # this function, and just call forward.
#    1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
#    1525         or _global_backward_pre_hooks or _global_backward_hooks
#    1526         or _global_forward_hooks or _global_forward_pre_hooks):
# -> 1527     return forward_call(*args, **kwargs)
#    1529 try:
#    1530     result = None
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/parallel/data_parallel.py:183, in DataParallel.forward(self, *inputs, **kwargs)
#     180     module_kwargs = ({},)
#     182 if len(self.device_ids) == 1:
# --> 183     return self.module(*inputs[0], **module_kwargs[0])
#     184 replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
#     185 outputs = self.parallel_apply(replicas, inputs, module_kwargs)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
#    1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
#    1517 else:
# -> 1518     return self._call_impl(*args, **kwargs)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1527, in Module._call_impl(self, *args, **kwargs)
#    1522 # If we don't have any hooks, we want to skip the rest of the logic in
#    1523 # this function, and just call forward.
#    1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
#    1525         or _global_backward_pre_hooks or _global_backward_hooks
#    1526         or _global_forward_hooks or _global_forward_pre_hooks):
# -> 1527     return forward_call(*args, **kwargs)
#    1529 try:
#    1530     result = None
#
# Cell In[1], line 140, in MemoryEfficientTCNModel.forward(self, x)
#     138 def forward(self, x):
#     139     for block in self.conv_blocks:
# --> 140         x = block(x)
#     142     x = self.attention(x)
#     144     x_gap = self.global_avg_pool(x).squeeze(-1)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
#    1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
#    1517 else:
# -> 1518     return self._call_impl(*args, **kwargs)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1527, in Module._call_impl(self, *args, **kwargs)
#    1522 # If we don't have any hooks, we want to skip the rest of the logic in
#    1523 # this function, and just call forward.
#    1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
#    1525         or _global_backward_pre_hooks or _global_backward_hooks
#    1526         or _global_forward_hooks or _global_forward_pre_hooks):
# -> 1527     return forward_call(*args, **kwargs)
#    1529 try:
#    1530     result = None
#
# Cell In[1], line 92, in GatedResidualNetwork.forward(self, x)
#      91 def forward(self, x):
# ---> 92     residual = self.residual(x)
#      93     x = self.conv1(x)
#      94     x = self.bn1(x)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
#    1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
#    1517 else:
# -> 1518     return self._call_impl(*args, **kwargs)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1527, in Module._call_impl(self, *args, **kwargs)
#    1522 # If we don't have any hooks, we want to skip the rest of the logic in
#    1523 # this function, and just call forward.
#    1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
#    1525         or _global_backward_pre_hooks or _global_backward_hooks
#    1526         or _global_forward_hooks or _global_forward_pre_hooks):
# -> 1527     return forward_call(*args, **kwargs)
#    1529 try:
#    1530     result = None
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/conv.py:310, in Conv1d.forward(self, input)
#     309 def forward(self, input: Tensor) -> Tensor:
# --> 310     return self._conv_forward(input, self.weight, self.bias)
#
# File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/conv.py:306, in Conv1d._conv_forward(self, input, weight, bias)
#     302 if self.padding_mode != 'zeros':
#     303     return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#     304                     weight, bias, self.stride,
#     305                     _single(0), self.dilation, self.groups)
# --> 306 return F.conv1d(input, weight, bias, self.stride,
#     307                 self.padding, self.dilation, self.groups)
#
# OutOfMemoryError: CUDA out of memory. Tried to allocate 36.70 GiB. GPU 0 has a total capacty of 15.89 GiB of which 14.94 GiB is free. Process 3055 has 964.00 MiB memory in use. Of the allocated memory 612.79 MiB is allocated by PyTorch, and 27.21 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF