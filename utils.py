import tensorflow as tf
import numpy as np


class OverfitProtection(tf.keras.callbacks.Callback):
    """ A callback to prevent overfitting in models. """
    def __init__(self, difference=0.1, patience=3, offset_start=3, verbose=True):
        self.difference = difference
        self.patience = patience
        self.offset_start = offset_start
        self.verbose = verbose
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        
        if epoch < self.offset_start:
            return

        epsilon = 1e-7
        ratio = loss / (val_loss + epsilon)

        if (1.0 - ratio) > self.difference:
            self.count += 1

            if self.verbose:
                print(f"Overfitting.. Patience: {self.count}/{self.patience}")

        elif self.count != 0:
            self.count -= 1
        
        if self.count >= self.patience:
            self.model.stop_training = True

            if self.verbose:
                print(f"Training stopped to prevent overfitting. Difference: {ratio}, Patience: {self.count}/{self.patience}")


def s2compress(arr, scale=10000.0):
    """ Compresses the values of an s2-array to a range of 0-2. """
    scaled = arr / scale
    limit = 65535.0 - scale

    bot = 1.0 + ((arr - scale) / limit)

    return np.where(arr >= scale, bot, scaled).astype("float32")


def s2compress_reverse(arr, scale=10000.0):
    top = arr * scale
    bot = 5.0 * ((11107.0 * arr) - 9107.0)

    return np.where(arr >= 1.0, bot, top)


def binary_metric():
    """ Returns a function that calculates the binary accuracy of a model. """

    def _binary_metric(y_true, y_pred):
        return tf.keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.01)

    _binary_metric.__name__ = "b_acc"

    return _binary_metric


def f1_loss(y_true, y_pred):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_pred * y_true, axis=0)
    fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)

    fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
    tn = tf.reduce_sum((1 - y_pred) * (1 - y_true), axis=0)

    soft_f1_class1 = (2 * tp) / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = (2 * tn) / (2*tn + fn + fp + 1e-16)

    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0

    macro_cost = tf.reduce_mean(cost) # average on all labels

    return macro_cost


def struct_mape_metric(beta=0.01):
    """ Returns a function that calculates the MAPE accuracy of building predictions, while ignoring the zero values. """
    def _struct_mape_metric(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        mask = tf.logical_or(y_true <= 0.0, y_pred <= 0.0)

        if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
            return 0.0

        num_top = tf.math.abs(tf.subtract(y_true, y_pred))
        num_bot = tf.math.add(tf.math.abs(y_true), beta)
        mape = tf.math.divide(num_top, num_bot) # beta mape

        masked = tf.boolean_mask(mape, mask)

        return tf.reduce_mean(masked)

    _struct_mape_metric.__name__ = "s_mape"

    return _struct_mape_metric


def beta_mape_loss(beta=0.01):
    def _mape_metric(y_true, y_pred):
        mape = tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + beta)))

        return mape

    _mape_metric.__name__ = "b_mape"

    return _mape_metric


def InceptionConvBlock(input_layer, filters, residual=False, activation="relu", kernel_initializer="glorot_normal"):
    """ Creates a convolutional block with a given number of layers. """
    if residual:
        filters = input_layer.shape[-1]

    conv1 = tf.keras.layers.Conv2D(filters, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv1 = tf.keras.layers.Conv2D(filters, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv1)

    conv2 = tf.keras.layers.Conv2D(filters, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv2 = tf.keras.layers.Conv2D(filters, 3, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv2)

    conv3 = tf.keras.layers.Conv2D(filters, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv3 = tf.keras.layers.Conv2D(filters, 5, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv3)

    merged = tf.keras.layers.Add()([conv1, conv2, conv3])
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Activation(activation)(merged)

    if residual:
        return tf.keras.layers.Add()([input_layer, merged])

    return merged


def ResNextConvBlock(
    input_layer,
    filters,
    kernel_size=3,
    cardinality=8,
    residual=False,
    activation="relu",
    kernel_initializer="glorot_normal",
):
    """ Creates a convolutional block with a given number of layers. """
    if residual:
        filters = input_layer.shape[-1]

    to_concatenate = []

    width = filters // cardinality

    for _ in range(cardinality):
        conv = tf.keras.layers.Conv2D(width, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
        conv = tf.keras.layers.Conv2D(width, kernel_size, padding="same", activation=activation, kernel_initializer=kernel_initializer)(conv)
        
        to_concatenate.append(conv)

    merged = tf.keras.layers.Concatenate()(to_concatenate)
    merged = tf.keras.layers.Conv2D(filters, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(merged)

    if residual:
        return tf.keras.layers.Add()([input_layer, merged])

    return merged


def SqueezeBlock(
    layer_input,
    ratio=8,
):
    """ Creates a squeeze block. """
    batch_size, _, _, channels = layer_input.shape
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(layer_input)
    excitation = tf.keras.layers.Dense(channels // ratio, activation="relu", use_bias=False)(squeeze)
    excitation = tf.keras.layers.Dense(channels, activation="sigmoid", use_bias=False)(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, channels])
    excitation = tf.keras.layers.Multiply()([layer_input, excitation])

    return excitation

def ReductionBlock(
    layer_input,
):
    """ Reduction Block for S2Pop-net. """
    track1 = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same",
    )(layer_input)

    return track1


def ExpansionBlock(
    layer_input,
    activation="relu",
    kernel_size=3,
    kernel_initializer="glorot_normal",
):
    """ Expansion block for S2Pop-net. """
    track1 = tf.keras.layers.Conv2DTranspose(
        layer_input.shape[-1],
        kernel_size=kernel_size,
        strides=(2, 2),
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(layer_input)

    return track1


def linear_fits(fits):
    """ Linearly interpolates between the given fits. """
    _fits = [x for x in fits]

    cur_sum = 0
    for idx, val in enumerate(_fits):
        if (idx + 1) > len(_fits) - 1:
            _fits[idx]["next_bs"] = _fits[idx]["bs"]
            _fits[idx]["next_lr"] = _fits[idx]["lr"]
        else:
            _fits[idx]["next_bs"] = _fits[idx + 1]["bs"]
            _fits[idx]["next_lr"] = _fits[idx + 1]["lr"]

        _fits[idx]["ie"] = cur_sum
        cur_sum += _fits[idx]["epochs"]

    linear = []
    for fit in _fits:
        steps = fit["epochs"]

        bs_start = fit["bs"]
        bs_end = fit["next_bs"]
        bs_dif = bs_end - bs_start
        bs_step = bs_dif / steps

        lr_start = fit["lr"]
        lr_end = fit["next_lr"]
        lr_dif = lr_end - lr_start
        lr_step = lr_dif / steps

        for epoch in range(fit["epochs"]):
            linear.append({
                "epoch_current": epoch + fit["ie"],
                "epoch_total": cur_sum,
                "bs": np.rint(bs_start + (epoch * bs_step),).astype(int),
                "lr": lr_start + (epoch * lr_step),
                "last": False if (epoch + 1) != fit["epochs"] else True,
            })
    
    return linear