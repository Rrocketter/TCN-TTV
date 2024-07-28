import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, \
    Concatenate, MultiHeadAttention, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def gated_residual_network(x, dilation_rate, nb_filters, kernel_size):
    prev_x = x
    x = LayerNormalization(epsilon=1e-6)(x)

    # Dilated causal convolution
    conv_out = Conv1D(filters=nb_filters * 2, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding='causal')(x)

    # Gating mechanism
    conv_out_a, conv_out_b = tf.split(conv_out, 2, axis=-1)
    x = tf.keras.activations.tanh(conv_out_a) * tf.keras.activations.sigmoid(conv_out_b)

    x = Conv1D(filters=nb_filters, kernel_size=1)(x)
    x = Dropout(0.1)(x)

    # Residual connection
    if prev_x.shape[-1] != nb_filters:
        prev_x = Conv1D(nb_filters, kernel_size=1)(prev_x)

    return Add()([prev_x, x])


def attention_block(x, nb_filters):
    attention_output = MultiHeadAttention(num_heads=4, key_dim=nb_filters)(x, x)
    x = Add()([x, attention_output])
    return LayerNormalization(epsilon=1e-6)(x)


def kepler_constraint_layer(x):
    # Implement Kepler's Third Law as a constraint
    # This is a simplified version and should be adapted based on your specific needs
    period = x[..., 0]
    semi_major_axis = x[..., 1]
    constrained = tf.pow(period, 2) - tf.pow(semi_major_axis, 3)
    return tf.concat([x, tf.expand_dims(constrained, -1)], axis=-1)


def create_novel_tcn_model(input_shape, nb_filters, kernel_size, nb_stacks, dilations, output_len):
    input_layer = Input(shape=input_shape)

    x = input_layer
    skip_connections = []

    for _ in range(nb_stacks):
        for dilation_rate in dilations:
            x = gated_residual_network(x, dilation_rate, nb_filters, kernel_size)
            skip_connections.append(x)

    if skip_connections:
        x = Add()(skip_connections)

    x = attention_block(x, nb_filters)

    x_gap = GlobalAveragePooling1D()(x)
    x_last = x[:, -1, :]
    x = Concatenate()([x_gap, x_last])

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)

    # Output both mean and standard deviation for uncertainty quantification
    mean_output = Dense(output_len, activation='linear', name='mean_output')(x)
    std_output = Dense(output_len, activation='softplus', name='std_output')(x)

    # Apply physical constraint
    combined_output = Concatenate()([mean_output, std_output])
    constrained_output = tf.keras.layers.Lambda(kepler_constraint_layer)(combined_output)

    model = Model(inputs=input_layer, outputs=constrained_output)
    return model


# Custom loss function incorporating physical constraints
def custom_loss(y_true, y_pred):
    mean_true, std_true = tf.split(y_true, 2, axis=-1)
    mean_pred, std_pred, constraint = tf.split(y_pred, 3, axis=-1)

    # Negative log likelihood loss
    nll = tfp.distributions.Normal(mean_pred, std_pred).log_prob(mean_true)
    nll_loss = -tf.reduce_mean(nll)

    # Physical constraint loss
    constraint_loss = tf.reduce_mean(tf.square(constraint))

    # Combine losses
    total_loss = nll_loss + 0.1 * constraint_loss
    return total_loss


# Generate synthetic TTV data
def generate_ttv_data(n_samples, n_timesteps):
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

# Reshape input data
X_train = X_train.reshape((train_split, n_timesteps, 1))
X_test = X_test.reshape((n_samples - train_split, n_timesteps, 1))

# Model parameters
input_shape = (n_timesteps, 1)
nb_filters = 64
kernel_size = 3
nb_stacks = 4
dilations = [1, 2, 4, 8, 16, 32, 64, 128]
output_len = 2  # Mean and standard deviation for TTV prediction

# Create and compile the model
model = create_novel_tcn_model(input_shape, nb_filters, kernel_size, nb_stacks, dilations, output_len)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=custom_loss,
              metrics=['mae'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions
predictions = model.predict(X_test)
mean_pred, std_pred, _ = np.split(predictions, 3, axis=-1)

# Visualize results
plt.figure(figsize=(12, 8))

# Plot training history
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MAE history
plt.subplot(2, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
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


# Implement a simple version of Integrated Gradients
def integrated_gradients(model, inputs, target_class_idx, m_steps=50, batch_size=32):
    baseline = np.zeros_like(inputs)

    alphas = np.linspace(0, 1, m_steps + 1)
    gradient_batches = []

    for alpha in alphas:
        interpolated_inputs = baseline + alpha * (inputs - baseline)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            predictions = model(interpolated_inputs)
            output = predictions[:, target_class_idx]

        gradients = tape.gradient(output, interpolated_inputs)
        gradient_batches.append(gradients)

    total_gradients = tf.reduce_mean(gradient_batches, axis=0)
    ig_attributions = (inputs - baseline) * total_gradients

    return ig_attributions


# Example usage of interpretability technique
sample_input = X_test[0:1]
ig_attributions = integrated_gradients(model, sample_input, target_class_idx=0)

# Visualize attributions
plt.figure(figsize=(12, 4))
plt.plot(ig_attributions[0].numpy().flatten())
plt.title('Feature Importance for TTV Prediction')
plt.xlabel('Time Step')
plt.ylabel('Attribution Score')
plt.show()