import tensorflow as tf
from utils import InceptionConvBlock, ReductionBlock, SqueezeBlock


def create_discriminator(
    INPUT_SHAPE,
    activation="relu",
    kernel_initializer="glorot_normal",
    name="discriminator",
    size=64,
    depth=1,
    squeeze=True,
    squeeze_ratio=16,
):
    """ Baseline building prediction network """
    SIZE_SMALL = size // 2
    SIZE_MEDIUM = size
    SIZE_LARGE = size + SIZE_SMALL

    model_input = tf.keras.Input(shape=(INPUT_SHAPE), name=f"{name}_input")

    # Outer-block
    conv = InceptionConvBlock(model_input, filters=SIZE_SMALL, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_SMALL, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(conv)

    # 32 x 32
    conv = InceptionConvBlock(redu, filters=SIZE_MEDIUM, residual=False, activation=activation, kernel_initializer=kernel_initializer)

    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_MEDIUM, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    
    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(redu)

    # 16 x 16
    conv = InceptionConvBlock(redu, filters=SIZE_LARGE, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    
    for _ in range(depth):
        conv = InceptionConvBlock(conv, filters=SIZE_LARGE, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(redu)

    average_pool = tf.keras.layers.GlobalAveragePooling2D()(redu)

    flat = tf.keras.layers.Dense(128, activation=activation, kernel_initializer=kernel_initializer)(average_pool)

    dropout = tf.keras.layers.Dropout(0.25)(flat)
    
    prediction = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, dtype="float32")(dropout)

    model = tf.keras.Model(
        inputs=model_input,
        outputs=prediction
    )

    return model
