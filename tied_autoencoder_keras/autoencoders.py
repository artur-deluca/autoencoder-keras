from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer
from tensorflow.keras.layers import Dense, Dropout


class DenseLayerAutoencoder(Dense):
    def __init__(self, layer_sizes, l2_normalize=False, dropout=0.0, *args, **kwargs):
        self.layer_sizes = layer_sizes
        self.l2_normalize = l2_normalize
        self.dropout = dropout
        self.kernels = []
        self.biases = []
        self.biases2 = []
        self.uses_learning_phase = True
        super().__init__(units=1, *args, **kwargs)  # 'units' not used

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        for i in range(len(self.layer_sizes)):

            self.kernels.append(
                self.add_weight(
                    shape=(
                        input_dim,
                        self.layer_sizes[i]),
                    initializer=self.kernel_initializer,
                    name='ae_kernel_{}'.format(i),
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(
                        shape=(
                            self.layer_sizes[i],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias_{}'.format(i),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))
            input_dim = self.layer_sizes[i]

        if self.use_bias:
            for n, i in enumerate(range(len(self.layer_sizes)-2, -1, -1)):
                self.biases2.append(
                    self.add_weight(
                        shape=(
                            self.layer_sizes[i],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias2_{}'.format(n),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))
            self.biases2.append(self.add_weight(
                        shape=(
                            input_shape[-1],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias2_{}'.format(len(self.layer_sizes)),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))

        self.built = True

    def call(self, inputs):
        return self.decode(self.encode(inputs))

    def _apply_dropout(self, inputs):
        dropped =  K.dropout(inputs, self.dropout)
        return K.in_train_phase(dropped, inputs)

    def encode(self, inputs):
        latent = inputs
        for i in range(len(self.layer_sizes)):
            if self.dropout > 0:
                latent = self._apply_dropout(latent)
            latent = K.dot(latent, self.kernels[i])
            if self.use_bias:
                latent = K.bias_add(latent, self.biases[i])
            if self.activation is not None:
                latent = self.activation(latent)
        if self.l2_normalize:
            latent = latent / K.l2_normalize(latent, axis=-1)
        return latent

    def decode(self, latent):
        recon = latent
        for i in range(len(self.layer_sizes)):
            if self.dropout > 0:
                recon = self._apply_dropout(recon)
            recon = K.dot(recon, K.transpose(self.kernels[len(self.layer_sizes) - i - 1]))
            if self.use_bias:
                recon = K.bias_add(recon, self.biases2[i])
            if self.activation is not None:
                recon = self.activation(recon)
        return recon

    def get_config(self):
        config = {
            'layer_sizes': self.layer_sizes
        }
        base_config = super().get_config()
        base_config.pop('units', None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)