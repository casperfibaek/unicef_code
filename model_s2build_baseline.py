import numpy as np
import tensorflow as tf
from utils import ConvBlock, ReductionBlock, ExpansionBlock, SqueezeBlock


EPSILON = 1.19e-07 # single precision flat epsilon


def create_model_baseline(
    input_shape,
    activation="relu",
    activation_output="relu",
    kernel_initializer="glorot_normal",
    name="baseline",
    size=128,
    depth=1,
    squeeze_ratio=8,
):
    """ Baseline building prediction network """
    SIZE_SMALL = size // 2
    SIZE_MEDIUM = size
    SIZE_LARGE = size + SIZE_SMALL

    model_input = tf.keras.Input(shape=(input_shape), name=f"{name}_input")

    # Outer-block
    conv = ConvBlock(model_input, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_skip_64 = ConvBlock(conv, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    conv = SqueezeBlock(conv_skip_64, channels=SIZE_SMALL, ratio=squeeze_ratio)
    redu = ReductionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 32 x 32
    conv = ConvBlock(redu, size=SIZE_MEDIUM, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_skip_32 = ConvBlock(conv, size=SIZE_MEDIUM, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    conv = SqueezeBlock(conv_skip_32, channels=SIZE_MEDIUM, ratio=squeeze_ratio)
    redu = ReductionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 16 x 16
    conv = ConvBlock(redu, size=SIZE_LARGE, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_skip_16 = ConvBlock(conv, size=SIZE_LARGE, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    conv = SqueezeBlock(conv_skip_16, channels=SIZE_LARGE, ratio=squeeze_ratio)
    redu = ReductionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 8 x 8
    conv = ConvBlock(redu, size=SIZE_LARGE, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv = SqueezeBlock(conv, channels=SIZE_LARGE, ratio=squeeze_ratio)
    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 16 x 16
    merg = tf.keras.layers.Concatenate()([expa, conv_skip_16])
    conv = ConvBlock(merg, size=SIZE_MEDIUM, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv = SqueezeBlock(conv, channels=SIZE_MEDIUM, ratio=squeeze_ratio)
    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 32 x 32
    merg = tf.keras.layers.Concatenate()([expa, conv_skip_32])
    conv = ConvBlock(merg, size=SIZE_MEDIUM, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv = SqueezeBlock(conv, channels=SIZE_MEDIUM, ratio=squeeze_ratio)
    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)
    
    # 64 x64
    merg = tf.keras.layers.Concatenate()([expa, conv_skip_64])

    # Main branch
    conv_main = ConvBlock(merg, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_main = ConvBlock(conv_main, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    conv_main = SqueezeBlock(conv_main, channels=SIZE_SMALL, ratio=squeeze_ratio)

    main_output = tf.keras.layers.Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation=activation_output,
        kernel_initializer=kernel_initializer,
        dtype="float32",
    )(conv_main)

    # Conf Branch
    conv_conf = ConvBlock(merg, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_conf = ConvBlock(conv_conf, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    conv_conf = SqueezeBlock(conv_conf, channels=SIZE_SMALL, ratio=squeeze_ratio)

    conf_output = tf.keras.layers.Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        kernel_initializer=kernel_initializer,
        dtype="float32",
    )(conv_conf)

    # Merge outputs
    concatenated_outputs = tf.keras.layers.Concatenate(axis=-1)([main_output, conf_output])

    model = tf.keras.Model(
        inputs=model_input,
        outputs=concatenated_outputs
    )

    return model


create_model_baseline((64, 64, 10)).summary()
