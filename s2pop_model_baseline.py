import tensorflow as tf
from utils import InceptionConvBlock, ReductionBlock, ExpansionBlock, SqueezeBlock


def create_model(
    input_shape,
    activation="swish",
    activation_output="relu",
    kernel_initializer="glorot_uniform",
    name="baseline",
    size=64,
    depth=1,
    squeeze_ratio=8,
    squeeze=False,
    dense_core=True,
):
    """ Baseline building prediction network """
    SIZE_SMALL = size // 2
    SIZE_MEDIUM = size
    SIZE_LARGE = size + SIZE_SMALL

    model_input = tf.keras.Input(shape=(input_shape), name=f"{name}_input")

    # Normalise input
    SCALE = 10000.0
    MAX_UINT16 = 65535.0
    LIMIT = MAX_UINT16 - SCALE

    cast_input = tf.cast(model_input, tf.float32)
    scaled_input = cast_input / SCALE
    bottom = 1.0 + ((cast_input - SCALE) / LIMIT)
    normalised_input = tf.where(cast_input >= SCALE, bottom, scaled_input)

    # Outer-block
    conv_skip_64 = InceptionConvBlock(normalised_input, filters=SIZE_SMALL, residual=False, activation=activation, kernel_initializer=kernel_initializer); conv = conv_skip_64
    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_SMALL, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(conv)

    # 32 x 32
    conv_skip_32 = InceptionConvBlock(redu, filters=SIZE_MEDIUM, residual=False, activation=activation, kernel_initializer=kernel_initializer); conv = conv_skip_32

    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_MEDIUM, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    
    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(redu)

    # 16 x 16
    conv_skip_16 = InceptionConvBlock(redu, filters=SIZE_LARGE, residual=False, activation=activation, kernel_initializer=kernel_initializer); conv = conv_skip_16
    
    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_LARGE, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(redu)

    # 8 x 8
    if dense_core:
        flatten = tf.keras.layers.Flatten()(redu)
        dense_core = tf.keras.layers.Dense(units=8 * 8 * 8, activation=activation, kernel_initializer=kernel_initializer)(flatten)
        conv = tf.keras.layers.Reshape((8, 8, 8))(dense_core)
        conv = InceptionConvBlock(conv, filters=SIZE_LARGE, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    else:
        conv = InceptionConvBlock(redu, filters=SIZE_LARGE, residual=False, activation=activation, kernel_initializer=kernel_initializer)

        if squeeze:
            conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 16 x 16
    merg = tf.keras.layers.Concatenate()([expa, conv_skip_16])

    conv = InceptionConvBlock(merg, filters=SIZE_MEDIUM, residual=False, activation=activation, kernel_initializer=kernel_initializer)

    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_MEDIUM, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 32 x 32
    merg = tf.keras.layers.Concatenate()([expa, conv_skip_32])

    conv = InceptionConvBlock(merg, filters=SIZE_MEDIUM, residual=False, activation=activation, kernel_initializer=kernel_initializer)

    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_MEDIUM, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    
    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)
    
    # 64 x64
    merg = tf.keras.layers.Concatenate()([expa, conv_skip_64])

    # Main branch
    conv_main = InceptionConvBlock(merg, filters=SIZE_SMALL, residual=False, activation=activation, kernel_initializer=kernel_initializer)

    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_SMALL, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    main_output = tf.keras.layers.Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation=activation_output,
        kernel_initializer=kernel_initializer,
        dtype="float32",
    )(conv_main)
    main_output = tf.clip_by_value(main_output, clip_value_min=0.0, clip_value_max=1.0)

    model = tf.keras.Model(
        inputs=model_input,
        outputs=main_output
    )

    return model
