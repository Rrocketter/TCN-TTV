import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class GatedResidualNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedResidualNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels * 2, kernel_size,
                               padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        # Apply LayerNorm across the channel dimension
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv1(x)
        x_a, x_b = torch.chunk(x, 2, dim=1)
        x = torch.tanh(x_a) * torch.sigmoid(x_b)
        x = self.conv2(x)
        x = self.dropout(x)
        return x + residual


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.
    """

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


class KeplerConstraintLayer(nn.Module):
    """
    Custom layer to enforce Kepler's Third Law as a soft constraint.
    """

    def forward(self, x):
        period, semi_major_axis = x[:, :2].chunk(2, dim=1)
        constrained = torch.pow(period, 2) - torch.pow(semi_major_axis, 3)
        return torch.cat([x, constrained], dim=1)


class NovelTCNModel(nn.Module):
    """
    Novel Temporal Convolutional Network (TCN) model for TTV detection and analysis.
    """

    def __init__(self, input_shape, nb_filters, kernel_size, nb_stacks, dilations, output_len):
        super(NovelTCNModel, self).__init__()
        self.conv_blocks = nn.ModuleList()
        self.nb_stacks = nb_stacks

        for _ in range(nb_stacks):
            for dilation in dilations:
                self.conv_blocks.append(GatedResidualNetwork(input_shape[0], nb_filters, kernel_size, dilation))

        self.attention = AttentionBlock(nb_filters)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(nb_filters * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean_output = nn.Linear(64, output_len)
        self.std_output = nn.Linear(64, output_len)

        self.kepler_constraint = KeplerConstraintLayer()

    def forward(self, x):
        skip_connections = []
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            skip_connections.append(x)

        x = torch.stack(skip_connections).sum(dim=0)
        x = self.attention(x)

        x_gap = self.global_avg_pool(x).squeeze(-1)
        x_last = x[:, :, -1]
        x = torch.cat([x_gap, x_last], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        mean_output = self.mean_output(x)
        std_output = torch.nn.functional.softplus(self.std_output(x))

        combined_output = torch.cat([mean_output, std_output], dim=1)
        constrained_output = self.kepler_constraint(combined_output)

        return constrained_output


def custom_loss(y_pred, y_true):
    """
    Custom loss function incorporating negative log likelihood and physical constraints.
    """
    mean_true, std_true = y_true.chunk(2, dim=1)
    mean_pred, std_pred, constraint = y_pred.chunk(3, dim=1)

    # Negative log likelihood loss
    nll = torch.distributions.Normal(mean_pred, std_pred).log_prob(mean_true)
    nll_loss = -torch.mean(nll)

    # Physical constraint loss
    constraint_loss = torch.mean(torch.square(constraint))

    # Combine losses
    total_loss = nll_loss + 0.1 * constraint_loss
    return total_loss


def generate_ttv_data(n_samples, n_timesteps):
    """
    Generate synthetic TTV data for training and testing.
    """
    time = np.linspace(0, 100, n_timesteps)
    period = 10.0
    amplitude = 0.1

    ttv = np.zeros((n_samples, n_timesteps))
    for i in range(n_samples):
        phase = np.random.uniform(0, 2 * np.pi)
        frequency = 1.0 / period
        ttv[i] = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        ttv[i] += np.random.normal(0, 0.01, n_timesteps)  # Add some noise

    return ttv


# Generate data
n_samples = 1000
n_timesteps = 500
X = generate_ttv_data(n_samples, n_timesteps)
y = np.column_stack((np.mean(X, axis=1), np.std(X, axis=1)))  # Use mean and std as target

# Split data
train_split = int(0.8 * n_samples)
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model parameters
input_shape = (1, n_timesteps)
nb_filters = 64
kernel_size = 3
nb_stacks = 4
dilations = [1, 2, 4, 8, 16, 32, 64, 128]
output_len = 2  # Mean and standard deviation for TTV prediction

# Create the model
model = NovelTCNModel(input_shape, nb_filters, kernel_size, nb_stacks, dilations, output_len)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = custom_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test.to(device))
        val_loss = custom_loss(val_outputs, y_test.to(device))
        val_losses.append(val_loss.item())

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test.to(device)).cpu().numpy()
mean_pred, std_pred, _ = np.split(predictions, 3, axis=1)

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

# Plot predicted vs true values
plt.subplot(2, 2, 3)
plt.scatter(y_test[:, 0], mean_pred, alpha=0.5)
plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
plt.title('Predicted vs True TTV Mean')
plt.xlabel('True TTV Mean')
plt.ylabel('Predicted TTV Mean')

# Plot uncertainty
plt.subplot(2, 2, 4)
plt.errorbar(range(len(mean_pred)), mean_pred.flatten(), yerr=std_pred.flatten(), fmt='o', alpha=0.5)
plt.title('Predictions with Uncertainty')
plt.xlabel('Sample Index')
plt.ylabel('TTV Prediction')

plt.tight_layout()
plt.show()


def integrated_gradients(model, inputs, target_class_idx, m_steps=50, batch_size=32):
    """
    Implement a simple version of Integrated Gradients for model interpretability.
    """
    baseline = torch.zeros_like(inputs)
    alphas = torch.linspace(0, 1, m_steps + 1).to(inputs.device)

    gradient_batches = []

    for alpha in alphas:
        interpolated_inputs = baseline + alpha * (inputs - baseline)
        interpolated_inputs.requires_grad_(True)

        predictions = model(interpolated_inputs)
        output = predictions[:, target_class_idx]

        gradients = torch.autograd.grad(outputs=output, inputs=interpolated_inputs)[0]
        gradient_batches.append(gradients)

    total_gradients = torch.stack(gradient_batches).mean(dim=0)
    ig_attributions = (inputs - baseline) * total_gradients

    return ig_attributions


# Example usage of interpretability technique
sample_input = X_test[:1].to(device)
ig_attributions = integrated_gradients(model, sample_input, target_class_idx=0)

# Visualize attributions
plt.figure(figsize=(12, 4))
plt.plot(ig_attributions[0, 0].cpu().detach().numpy())
plt.title('Feature Importance for TTV Prediction')
plt.xlabel('Time Step')
plt.ylabel('Attribution Score')
plt.show()