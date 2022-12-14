import os
import numpy as np
import buteo as beo
import tensorflow as tf
from glob import glob
from utils import OverfitProtection, struct_mape_metric, create_GAN_data, f1_loss
from s2pop_predict import predict
from s2pop_model_baseline import create_model
from s2pop_model_discriminator import create_discriminator
from s2super.super_sample import super_sample_patches, get_s2super_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all messages are logged (default behavior), 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed
tf.get_logger().setLevel("ERROR")

DATA_FOLDER = "D:/CFI/data/sen2pop_v2/"
MODEL_FOLDER = "./models/"

BASE_LR = 1e-4
VAL_BS = 256
INPUT_SHAPE = (64, 64, 10)
ACTIVATION = "relu"
ACTIVATION_OUTPUT = "relu"
KERNEL_INITIALIZER = "glorot_uniform"
NAME = "s2pop_gan_v20"
SIZE = 128
SQUEEZE_RATIO = 8
DENSE_CORE = True
SQUEEZE = True
DEPTH = 2
OPTIMIZER_PRED = "adam"
OPTIMIZER_DISC = "adam"
BETA = 0.01
METRICS = ["mse", struct_mape_metric(BETA), tf.keras.metrics.Precision(BETA), tf.keras.metrics.Recall(BETA)]
PREDICT = True
PREDICT_CALLBACK = True
COMPILED = False

LIMIT = 50000
VAL_SPLIT = 0.1
TARGET = 0 # Buildings = 0, Roads = 1, 

# ------------------------------------ LOAD DATA ------------------------------------ #
x_train = np.load(DATA_FOLDER + "x_train_50000.npz")["data"]
x_test = np.load(DATA_FOLDER + "x_test_5000.npz")["data"]

# x_train = np.load(DATA_FOLDER + "x_train_50000_sharp.npz")["data"]
# x_test = np.load(DATA_FOLDER + "x_test_5000_sharp.npz")["data"]

y_train = np.load(DATA_FOLDER + "y_train_50000.npz")["label"][:, :, :, TARGET][:, :, :, np.newaxis]
y_test = np.load(DATA_FOLDER + "y_test_5000.npz")["label"][:, :, :, TARGET][:, :, :, np.newaxis]

x_train = x_train[:LIMIT, :, :, :]
y_train = y_train[:LIMIT, :, :, :]

val_total = int(x_train.shape[0] * VAL_SPLIT)

x_val = x_train[-val_total:, :, :, :]; x_train = x_train[:-val_total, :, :, :]
y_val = y_train[-val_total:, :, :, :]; y_train = y_train[:-val_total, :, :, :]

zero_mask_train = y_train.sum(axis=(1, 2, 3)) != 0.0
y_train_reduced = y_train[zero_mask_train]
x_train_reduced = x_train[zero_mask_train]

zero_mask_val = y_val.sum(axis=(1, 2, 3)) != 0.0
y_val_reduced = y_val[zero_mask_val]
x_val_reduced = x_val[zero_mask_val]

epochs_discriminator = 5
fits = [
    { "epochs": 25, "bs":  8,  "lr": 1e-04},
    { "epochs": 20, "bs": 16,  "lr": 1e-04},
    { "epochs": 15, "bs": 32,  "lr": 1e-04},
    { "epochs": 10, "bs": 64,  "lr": 1e-04},
    { "epochs": 10, "bs": 96,  "lr": 1e-04},
    { "epochs": 5, "bs": 96,  "lr": 1e-05},
    { "epochs": 5, "bs": 96,  "lr": 1e-05},
    { "epochs": 5, "bs": 96,  "lr": 1e-05},
    { "epochs": 5, "bs": 96,  "lr": 1e-05},
    { "epochs": 5, "bs": 96,  "lr": 1e-05},
]

cur_sum = 0
for nr, val in enumerate(fits):
    fits[nr]["ie"] = cur_sum
    cur_sum += fits[nr]["epochs"]

# ------------------------------------ MODEL DESIGN ------------------------------------ #
model_s2pop = create_model(
    INPUT_SHAPE,
    activation=ACTIVATION,
    activation_output=ACTIVATION_OUTPUT,
    kernel_initializer=KERNEL_INITIALIZER,
    name=NAME,
    size=SIZE,
    squeeze_ratio=SQUEEZE_RATIO,
    squeeze=SQUEEZE,
    depth=DEPTH,
    dense_core=DENSE_CORE,
)
model_s2pop.summary()

model_discriminator = create_discriminator(
    model_s2pop.output.shape[1:],
    activation="relu",
    kernel_initializer="glorot_normal",
    name="discriminator",
    size=64,
    depth=1,
    squeeze=True,
    squeeze_ratio=8,
)

tf.keras.utils.plot_model(model_s2pop, to_file=f"./{NAME}_GENERATOR.png")
tf.keras.utils.plot_model(model_discriminator, to_file=f"./{NAME}_DISCRIMINATOR.png")

@tf.function(jit_compile=COMPILED)
def gan_loss(_y_true, y_pred):
    STDDEV = 0.001

    random_k = tf.random.uniform([1], minval=0, maxval=4, dtype=tf.dtypes.int32)
    y_pred_rot = tf.image.rot90(y_pred, k=random_k[0])

    y_pred_noise = tf.random.normal(shape=tf.shape(y_pred), mean=0.0, stddev=STDDEV, dtype=tf.float32)
    y_pred_adj = y_pred_rot + y_pred_noise

    discriminator_pred = model_discriminator(y_pred_adj)

    gan_loss = tf.math.reduce_mean(discriminator_pred)

    return gan_loss

gan_loss.__name__ = "gan_loss"

@tf.function(jit_compile=COMPILED)
def test_loss(y_true, y_pred):
    loss_mse = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
    loss_gan = gan_loss(y_true, y_pred)
    loss_f1 = f1_loss(y_true, y_pred)

    return (loss_mse + (loss_gan * 0.0025) + (loss_f1 * 0.0025))

test_loss.__name__ = "test_loss"

@tf.function(jit_compile=COMPILED)
def prop(y_true, y_pred):
    loss_mse = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
    loss_gan = gan_loss(y_true, y_pred)
    loss_f1 = f1_loss(y_true, y_pred)

    return loss_mse / (loss_mse + (loss_gan * 0.0025) + (loss_f1 * 0.0025))

prop.__name__ = "prop"


METRICS.append(gan_loss)
METRICS.append(prop)

model_s2pop.compile(optimizer=OPTIMIZER_PRED, loss=test_loss, metrics=METRICS, jit_compile=COMPILED)
model_discriminator.compile(optimizer=OPTIMIZER_DISC, loss="binary_crossentropy", metrics=["accuracy"], jit_compile=COMPILED)

def image_snapshop(epoch, model):
    images = glob("D:/CFI/predictions/test_images/*.tif")
    # images = glob("D:/CFI/predictions/test_images/north-america_10.tif")
    for img in images:
        img_name = os.path.splitext(os.path.basename(img))[0]
        
        if len(img_name.split("_")) > 2:
            continue
        
        arr = beo.raster_to_array(img)[:, :, 1:] # loose scl
        predicted = predict(model, arr, arr, merge_method="mean")

        outname = f"D:/CFI/predictions/{NAME}_{img_name}_{epoch + 1}_pred.tif"
        beo.array_to_raster(predicted, reference=img, out_path=outname)

class PredictCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        image_snapshop(epoch, self.model)

image_snapshop(-1, model_s2pop)

def batch_generator_train(batchsize):
    global x_train, y_train
    patches_len = y_train.shape[0]
    idx = 0

    while True:
        x_batch = x_train[idx:idx + batchsize]
        y_batch = y_train[idx:idx + batchsize]

        # Random rotation
        rotation = np.random.randint(0, 4)
        if rotation != 0:
            x_batch = np.rot90(x_batch, axes=(1, 2))
            y_batch = np.rot90(y_batch, axes=(1, 2))

        # Random flips
        flips =  np.random.randint(0, 4)
        if flips == 3:
            x_batch = np.flip(x_batch, axis=(1, 2))
            y_batch = np.flip(y_batch, axis=(1, 2))
        elif flips == 2:
            x_batch = np.flip(x_batch, axis=(2))
            y_batch = np.flip(y_batch, axis=(2))
        elif flips == 1:
            x_batch = np.flip(x_batch, axis=(1))
            y_batch = np.flip(y_batch, axis=(1))
        else:
            pass

        # Normal noise
        if np.random.randint(0, 2) == 1:
            noise = np.random.normal(0, 0.001, x_batch.shape)
            x_batch += noise

        # Random scale
        if np.random.randint(0, 2) == 1:
            scale = np.random.normal(0, 0.005)
            x_batch += scale
            x_batch = np.clip(x_batch, 0.0, 2.0)

        yield x_batch, y_batch

        idx = idx + batchsize

        if idx + batchsize > patches_len:
            idx = 0
            mask = np.random.permutation(len(y_train))
            x_train = x_train[mask]
            y_train = y_train[mask]
        
# ------------------------------------ START LOOP ------------------------------------------- #
for idx, fit in enumerate(fits):
    use_epoch = fit["epochs"]
    use_ie = fit["ie"]
    use_bs = fit["bs"]
    use_lr = fit["lr"]

    model_s2pop.optimizer.lr.assign(use_lr)

    # mask_disc = np.random.permutation(x_train.shape[0])
    # x_train = x_train[mask_disc]
    # y_train = y_train[mask_disc]

    callbacks = [OverfitProtection(patience=3, difference=0.20, offset_start=3)]
    if PREDICT_CALLBACK:
        callbacks.append(PredictCallback())

    model_s2pop.fit(
        x=batch_generator_train(use_bs),
        # x=x_train,
        # y=y_train,
        validation_data=(x_val, y_val),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        batch_size=use_bs,
        steps_per_epoch=int(y_train.shape[0] / use_bs),
        validation_batch_size=VAL_BS,
        use_multiprocessing=True,
        workers=0,
        verbose=1,
        callbacks=callbacks
    )

    print("Test data:")
    model_s2pop.evaluate(x_test, y_test)

    if idx + 1 == len(fits):
        break

    model_discriminator.optimizer.lr.assign(use_lr)

    d_x_train, d_y_train = create_GAN_data(x_train_reduced, y_train_reduced, model_s2pop, VAL_BS)
    d_x_val, d_y_val = create_GAN_data(x_val_reduced, y_val_reduced, model_s2pop, VAL_BS)
    model_discriminator.fit(
        x=d_x_train,
        y=d_y_train,
        validation_data=(d_x_val, d_y_val),
        shuffle=True,
        epochs=epochs_discriminator + use_ie,
        initial_epoch=use_ie,
        batch_size=use_bs,
        validation_batch_size=VAL_BS,
        use_multiprocessing=True,
        workers=0,
        verbose=1,
        callbacks=[
            OverfitProtection(
                patience=3,
                difference=0.2, # 20% overfit allowed
                offset_start=3, # disregard overfit for the first epoch
            ),
        ],
    )

model_s2pop.save(f"D:/CFI/models/{NAME}.model")

if PREDICT:
    images = glob("D:/CFI/predictions/test_images/*.tif")
    for img in images:
        img_name = os.path.splitext(os.path.basename(img))[0]
        
        if len(img_name.split("_")) > 2:
            continue
        
        arr = beo.raster_to_array(img)[:, :, 1:] # loose scl
        predicted = predict(model_s2pop, arr, arr)

        outname = f"D:/CFI/predictions/{img_name}_{NAME}_pred.tif"
        beo.array_to_raster(predicted, reference=img, out_path=outname)

        print(f"Generated outname: {outname}")
