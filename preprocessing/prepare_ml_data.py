import numpy as np
from sklearn.model_selection import train_test_split


def load_processed_data(file_path):
    """Load processed data from .npz file."""
    with np.load(file_path) as data:
        return [data[arr] for arr in data.files]


def pad_or_truncate(array, target_len):
    """Pad or truncate array to target length."""
    if len(array) > target_len:
        return array[:target_len]
    else:
        return np.pad(array, (0, target_len - len(array)), mode='constant')


def prepare_data_for_ml(processed_data_dir, window_size=1000, test_size=0.2, batch_size=32):
    """Prepare data for ML model."""
    all_transits = []
    all_labels = []  # You'll need to define how to label your data

    # Load and format all processed data
    for file in os.listdir(processed_data_dir):
        if file.endswith('.npz'):
            file_path = os.path.join(processed_data_dir, file)
            transit_windows = load_processed_data(file_path)

            for time, flux in transit_windows:
                formatted_flux = pad_or_truncate(flux, window_size)
                all_transits.append(formatted_flux)

                # Add labels here. For example:
                # all_labels.append(1 if 'planet' in file else 0)

    # Convert to numpy arrays
    X = np.array(all_transits)
    y = np.array(all_labels)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create batches
    def batch_generator(X, y, batch_size):
        while True:
            for i in range(0, len(X), batch_size):
                yield X[i:i + batch_size], y[i:i + batch_size]

    train_generator = batch_generator(X_train, y_train, batch_size)
    test_generator = batch_generator(X_test, y_test, batch_size)

    return train_generator, test_generator, X_train.shape[0], X_test.shape[0]


if __name__ == "__main__":
    processed_data_dir = "path/to/processed/data"
    train_gen, test_gen, train_steps, test_steps = prepare_data_for_ml(processed_data_dir)

    # Now you can use train_gen and test_gen with your ML model
    # For example, with a Keras model:
    # model.fit(train_gen, steps_per_epoch=train_steps//batch_size, ...)