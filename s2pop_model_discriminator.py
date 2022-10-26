import tensorflow as tf
from utils import ResNextConvBlock, ReductionBlock, ExpansionBlock, SqueezeBlock


def create_discriminator(
    INPUT_SHAPE,
    activation="relu",
    kernel_initializer="glorot_normal",
    name="discriminator",
    size=64,
    depth=1,
    cardinality=4,
    squeeze=True,
    squeeze_ratio=16,
):
    """ Baseline building prediction network """
    SIZE_SMALL = size // 2
    SIZE_MEDIUM = size
    SIZE_LARGE = size + SIZE_SMALL

    model_input = tf.keras.Input(shape=(INPUT_SHAPE), name=f"{name}_input")

    # Outer-block
    conv = ResNextConvBlock(model_input, filters=SIZE_SMALL, cardinality=cardinality, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    for _ in range(depth):
        conv = ResNextConvBlock(conv, filters=SIZE_SMALL, cardinality=cardinality, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(conv)

    # 32 x 32
    conv = ResNextConvBlock(redu, filters=SIZE_MEDIUM, cardinality=cardinality, residual=False, activation=activation, kernel_initializer=kernel_initializer)

    for _ in range(depth):
        conv = ResNextConvBlock(conv, filters=SIZE_MEDIUM, cardinality=cardinality, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    
    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(redu)

    # 16 x 16
    conv = ResNextConvBlock(redu, filters=SIZE_LARGE, cardinality=cardinality, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    
    for _ in range(depth):
        conv = ResNextConvBlock(conv, filters=SIZE_LARGE, cardinality=cardinality, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    if squeeze:
        conv = SqueezeBlock(conv, ratio=squeeze_ratio)

    redu = ReductionBlock(redu)

    average_pool = tf.keras.layers.GlobalAveragePooling2D()(redu)

    flat = tf.keras.layers.Dense(128, activation=activation, kernel_initializer=kernel_initializer)(average_pool)
    
    prediction = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, dtype="float32")(flat)

    model = tf.keras.Model(
        inputs=model_input,
        outputs=prediction
    )

    return model
