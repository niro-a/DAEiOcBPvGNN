# Adapted from Nolle et al. (2022)
# Url: https://github.com/tnolle/binet/blob/master/april/anomalydetection/autoencoder.py


class AE():

    name = 'AE'

    config = dict(hidden_layers=2,
                  hidden_size_factor=.2,
                  noise=None)

    def __init__(self):
        """Initialize DAE model."""
        self.model = None

    @staticmethod
    def model_fn(dataset, **kwargs):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import Adam

        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')

        features = dataset.flat_onehot_features_2d

        # Parameters
        input_size = features.shape[1]

        # Input layer
        input = Input(shape=(input_size,), name='input')
        x = input

        # Noise layer
        if noise is not None:
            x = GaussianNoise(noise)(x)

        # Hidden layers
        for i in range(hidden_layers):
            if isinstance(hidden_size_factor, list):
                factor = hidden_size_factor[i]
            else:
                factor = hidden_size_factor
            x = Dense(int(input_size * factor), activation='relu', name=f'hid{i + 1}')(x)
            x = Dropout(0.5)(x)

        # Output layer
        output = Dense(input_size, activation='linear', name='output')(x)

        # Build model
        model = Model(inputs=input, outputs=output)

        # Compile model
        model.compile(
            optimizer=Adam(lr=0.0001, beta_2=0.99),
            loss='mean_squared_error',
        )

        return model, features, features  # Features are also targets

    def detect(self, dataset):
        """
        Calculate the anomaly score for each event attribute in each trace.
        Anomaly score here is the mean squared error.

        :param traces: traces to predict
        :return:
            scores: anomaly scores for each attribute;
                            shape is (#traces, max_trace_length - 1, #attributes)

        """
        import numpy as np

        # Get features
        _, features, _ = self.model_fn(dataset, **self.config)

        # Parameters
        input_size = int(self.model.input.shape[1])
        features_size = int(features.shape[1])
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)], mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        # Get predictions
        predictions = self.model.predict(features)

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