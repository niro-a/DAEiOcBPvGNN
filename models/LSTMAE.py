# Adapted from Nguyen et al. (2019)
# Url: https://github.com/IELunist/Autoencoders-for-Improving-Quality-of-Process-Event-Logs/blob/master/multivariate-anomaly-detection-for-event-logs-master/experiment/LSTM-AE-Keras.ipynb

from keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed
from keras.models import Model
from types import SimpleNamespace
import numpy as np


class LSTMAE():
    """
    Creates an LSTM Autoencoder (VAE). Returns Autoencoder, Encoder, Generator.
    (All code by fchollet - see reference.)
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        latent_dim: int, latent z-layer shape.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
    """

    parser = {
        'batch_size': 8192,
        'epochs': 100,
        'no_cuda': False,
        'seed': 7,
        'layer1': 1000,
        'layer2': 100,
        'lr': 0.002,
        'betas': (0.9, 0.999),
        'lr_decay': 0.90,
    }

    args = SimpleNamespace(**parser)

    def __init__(self):
        """Initialize LSTMAE model."""
        self.model = None

    def model_fn(self, dataset):
        timesteps = int(dataset.max_len)
        input_dim = sum(dataset.attribute_dims)
        latent_dim = 100
        features = dataset.flat_onehot_features_2d

        # Reshape the features to the correct LSTM input shape
        features = features.values.reshape((-1, timesteps, input_dim))

        inputs = Input(shape=(timesteps, input_dim,))
        encoded = LSTM(latent_dim)(inputs)

        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        # Add a TimeDistributed layer with a Dense unit
        decoded = TimeDistributed(Dense(input_dim))(decoded)

        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)

        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mae')

        return autoencoder, features, features

    def detect(self, dataset):
        """
        Calculate the anomaly score for each event attribute in each trace.
        Anomaly score here is the mean squared error.

        :param traces: traces to predict
        :return:
            scores: anomaly scores for each attribute;
                            shape is (#traces, max_trace_length - 1, #attributes)

        """
        # Get features
        autoencoder, features, _ = self.model_fn(dataset)

        # Train the autoencoder
        autoencoder.fit(features, features, epochs=self.args.epochs, batch_size=self.args.batch_size, verbose=False)

        # Get predictions
        predictions = autoencoder.predict(features, batch_size=1024)

        features = features.reshape(features.shape[0], -1)
        predictions = predictions.reshape(predictions.shape[0], -1)

        # Calculate error
        errors = np.power(features - predictions, 2)

        # Split the errors according to the attribute dims
        split = np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
        errors = np.split(errors, split, axis=1)
        errors = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors])

        # Init anomaly scores array
        scores = np.zeros((len(dataset.y), dataset.max_len, len(dataset.attribute_dims)))

        for i in range(len(dataset.attribute_dims)):
            error = errors[i::len(dataset.attribute_dims)]
            scores[:, :, i] = error.T

        # Number of groups
        num_groups = int(dataset.mask.shape[1] / sum(dataset.attribute_dims))

        # Initialize an empty mask
        grouped_mask = np.zeros((dataset.mask.shape[0], num_groups), dtype=bool)

        # Populate the mask
        for i in range(num_groups):
            start = i * sum(dataset.attribute_dims)
            end = (i + 1) * sum(dataset.attribute_dims)
            grouped_mask[:, i] = dataset.mask.iloc[:, start:end].any(axis=1)

        # Compute the mean along the third dimension of scores
        mean_scores = np.mean(scores, axis=2)

        mean_scores = mean_scores.flatten()
        grouped_mask = grouped_mask.flatten()

        mean_scores = mean_scores[~grouped_mask]

        return mean_scores