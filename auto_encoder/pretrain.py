from os.path import join

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from tied_autoencoder_keras import DenseLayerAutoencoder

NOISE_LEVEL = 0.1

def get_embedder(model, i, opt):
    """Get a model to extract inermediate embeddings from a neural net.

    Args:
        model (keras Model instance) - The neural network model
        i (int) - The embedding that will be outputted will be from the (i-1)th layer in model.
            In other words, it will extract the input to the ith layer of model.
        opt (string or Optimizer instance) - The optimization algorithm to use during training
    Returns:
        embedder (keras Model instance) - A neural network that will output an intermediate
            embedding from the original model.
    """
    embedder = Model(inputs=model.layers[0].input, outputs=model.layers[i-1].output)
    embedder.compile(loss='mean_squared_error', optimizer=opt)
    return embedder


def pretrain_model(model, input_dim, opt, train_X, act, early_stop=True, epochs=1000, batch_size=64, valid_split=0.2):
    """Use greedy layerwise pretraining to initialize the weights of a neural net.

    Iterates through the layers of model (ignores layers that are not Dense layers),
    and pretrains each one, bottom up, using the greedy layerwise pretraining strategy
    described in Vincent et al. (http://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf).
    The 'Greedy module' used is a tied-weights autoencoder.

    Args:
        model (keras Model instance) - The neural network model to be pretrained
        input_dim (int) - Number of dimensions in the input data
        opt (string or Optimizer instance) - The optimization algorithm to use during training
        train_X (np.ndarray) - The input data to use for pretraining
        act (str) - Activation function to use in greedy module autoencoders (should be the same
            as what you use in model)
        early_stop (bool) - If True, will do early stopping after training has converged
            when training each autoencoder
        epochs (int) - Maximum number of epochs to train each autoencoder for
        batch_size (int) - Batch size to use for training each autoencoder
        valid_split (float) - Float between 0 and 1. Fraction of the training data to be used as validation data. 
    """
    for i in range(1, len(model.layers)):
        print("Pretraining layer {}".format(i))
        if not isinstance(model.layers[i], Dense):
            print("layer {}:{} is not a Dense layer, skipping".format(i, type(model.layers[i])))
            continue
        # Get the input data to train on. Need to feed forward the original X data to get it embedded as the i-th
        # layer's input, and also need to get a noisy version of it. Autoencoder will take in the noisy version
        # of the embedded data, and try to reconstruct the clean embedded data.
        embedder = get_embedder(model, i, opt)
        embedded_data = embedder.predict(train_X)
        corrupted = embedded_data + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=embedded_data.shape)
        
        # construct the greedy module (a tied-weights denoising autoencoder)
        inputs = Input(shape=(embedded_data.shape[1],))
        x = DenseLayerAutoencoder([model.layers[i].output_shape[-1]], activation=act)(inputs)
        dae = Model(inputs=inputs, outputs=x)
        
        print("DAE architecture:")
        print(dae.summary())
        dae.compile(loss='mean_squared_error', optimizer=opt)
        callbacks_list = []
        if early_stop:
            print("using early stopping")
            callbacks_list = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    verbose=1)]
        dae.fit(
            corrupted,
            embedded_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=valid_split,
            callbacks=callbacks_list)

        # After training the denoising autoencoder, take its weights and place them in the
        # corresponding layer in your original model. Pretraining that layer is now complete.
        model.layers[i].set_weights(dae.layers[1].get_weights()[:2])
        model.layers[i].trainable = False
    
    for layer in model.layers:
        layer.trainable = True

    return model
