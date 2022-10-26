import os
import numpy as np
import buteo as beo
import tensorflow as tf
from glob import glob
from utils import OverfitProtection, struct_mape_metric, f1_loss
from s2pop_predict import predict
from s2pop_model_baseline import create_model
from s2pop_model_discriminator import create_discriminator

DATA_FOLDER = "D:/CFI/data/sen2pop_v2/"
MODEL_FOLDER = "./models/"

BASE_LR = 1e-4
VAL_BS = 256
INPUT_SHAPE = (64, 64, 10)
ACTIVATION = "swish"
ACTIVATION_OUTPUT = "relu"
KERNEL_INITIALIZER = "glorot_uniform"
NAME = "s2pop_gan_v4"
SIZE = 64
CARDINALITY = 1
SQUEEZE_RATIO = 8
DENSE_CORE = False
SQUEEZE = False
DEPTH = 1
OPTIMIZER_PRED = tf.optimizers.Adam(learning_rate=BASE_LR)
OPTIMIZER_DISC = tf.optimizers.Adam(learning_rate=BASE_LR)
LOSS = "logcosh"
BETA = 0.01
METRICS = ["mse", "mae", struct_mape_metric(BETA), tf.keras.metrics.Precision(BETA), tf.keras.metrics.Recall(BETA)]
PREDICT = True

LIMIT = 20000
VAL_SPLIT = 0.1
TARGET = 0 # Buildings = 0, Roads = 1, 

# ------------------------------------ LOAD DATA ------------------------------------ #
x_train = np.load(DATA_FOLDER + "x_train_50000.npz")["data"]
x_test = np.load(DATA_FOLDER + "x_test_5000.npz")["data"]

y_train = np.load(DATA_FOLDER + "y_train_50000.npz")["label"][:, :, :, TARGET][:, :, :, np.newaxis]
y_test = np.load(DATA_FOLDER + "y_test_5000.npz")["label"][:, :, :, TARGET][:, :, :, np.newaxis]

x_train = x_train[:LIMIT, :, :, :]
y_train = y_train[:LIMIT, :, :, :]

val_total = int(x_train.shape[0] * VAL_SPLIT)

x_val = x_train[-val_total:, :, :, :]; x_train = x_train[:-val_total, :, :, :]
y_val = y_train[-val_total:, :, :, :]; y_train = y_train[:-val_total, :, :, :]

fits = [
    { "epochs": 10, "bs": 64,  "lr": 1e-04},
    { "epochs": 10, "bs": 80,  "lr": 1e-04},
    { "epochs": 10, "bs": 96,  "lr": 1e-04},
    { "epochs": 10, "bs": 128, "lr": 1e-04},
]

cur_sum = 0
for nr, val in enumerate(fits):
    fits[nr]["ie"] = cur_sum
    cur_sum += fits[nr]["epochs"]

# ------------------------------------ MODEL DESIGN ------------------------------------ #
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, _epoch, logs=None):
        global save_best_model, best_val_loss
        val_loss = logs['val_loss']

        if val_loss <= best_val_loss:
            save_best_model.set_weights(self.model.get_weights())

model_s2pop = create_model(
    INPUT_SHAPE,
    activation=ACTIVATION,
    activation_output=ACTIVATION_OUTPUT,
    kernel_initializer=KERNEL_INITIALIZER,
    name=NAME,
    size=SIZE,
    cardinality=CARDINALITY,
    squeeze_ratio=SQUEEZE_RATIO,
    squeeze=SQUEEZE,
    depth=DEPTH,
    dense_core=DENSE_CORE,
)

model_discriminator = create_discriminator(
    model_s2pop.output.shape[1:],
    activation="relu",
    kernel_initializer="glorot_normal",
    name="discriminator",
    size=64,
    depth=1,
    cardinality=4,
    squeeze=True,
    squeeze_ratio=16,
)

save_best_model = create_model(
    INPUT_SHAPE,
    activation=ACTIVATION,
    activation_output=ACTIVATION_OUTPUT,
    kernel_initializer=KERNEL_INITIALIZER,
    name=NAME,
    size=SIZE,
    cardinality=CARDINALITY,
    squeeze_ratio=SQUEEZE_RATIO,
    squeeze=SQUEEZE,
    depth=DEPTH,
    dense_core=DENSE_CORE,
)
save_best_model.set_weights(model_s2pop.get_weights())

def gan_loss(y_true, y_pred):
    STDDEV = 0.001

    random_k = tf.random.uniform([1], minval=0, maxval=4, dtype=tf.dtypes.int32)
    y_pred_rot = tf.image.rot90(y_pred, k=random_k[0])

    y_pred_noise = tf.random.normal(shape=tf.shape(y_pred), mean=0.0, stddev=STDDEV, dtype=tf.float32)
    y_pred_adj = y_pred_rot + y_pred_noise

    discriminator_pred = model_discriminator(y_pred_adj)

    gan_loss = tf.math.reduce_mean(discriminator_pred)

    return gan_loss

gan_loss.__name__ = "gan_loss"

def test_loss(y_true, y_pred):
    true_avg = tf.math.pow(tf.math.reduce_mean(y_true) + 1.0, 2.0)
    loss_f1 = f1_loss(y_true, y_pred)
    loss_gan = gan_loss(y_true, y_pred)

    loss_logcosh = tf.math.reduce_mean(tf.keras.losses.logcosh(y_true, y_pred))

    # return (loss_logcosh + (loss_gan * 0.001)) * true_avg
    return ((loss_logcosh * 1.0) + (loss_f1 * 0.1) + (loss_gan * 0.01)) * true_avg


test_loss.__name__ = "test_loss"

def mean_pred_metric(y_true, y_pred):
    return tf.math.reduce_mean(y_pred)

mean_pred_metric.__name__ = "mean_pred_metric"

def weighted_logcosh_loss(y_true, y_pred):
    true_avg = tf.math.pow(tf.math.reduce_mean(y_true) + 1.0, 2.0)
    loss_logcosh = tf.math.reduce_mean(tf.keras.losses.logcosh(y_true, y_pred)) * true_avg

    return loss_logcosh

weighted_logcosh_loss.__name__ = "w_logcosh"

METRICS.append(gan_loss)
METRICS.append(test_loss)
METRICS.append(mean_pred_metric)

model_s2pop.compile(optimizer=OPTIMIZER_PRED, loss=weighted_logcosh_loss, metrics=METRICS)
model_discriminator.compile(optimizer=OPTIMIZER_DISC, loss="binary_crossentropy", metrics=["accuracy"])
save_best_model.compile(optimizer=OPTIMIZER_PRED, loss=LOSS, metrics=METRICS)

# ------------------------------------ PRERUN PRERUN PRERUN ------------------------------------ #
GENERATOR_PRERUN_FITS = 10
GENERATOR_PRERUN_LR = 1e-4
GENERATOR_PRERUN_BS = 64

DISCRIMINATOR_PRERUN_FITS = 10
DISCRIMINATOR_PRERUN_LR = 1e-4
DISCRIMINATOR_PRERUN_BS = 64

model_s2pop.optimizer.lr.assign(GENERATOR_PRERUN_LR)

model_s2pop.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    shuffle=True,
    epochs=GENERATOR_PRERUN_FITS,
    initial_epoch=0,
    batch_size=GENERATOR_PRERUN_BS,
    validation_batch_size=VAL_BS,
    use_multiprocessing=True,
    workers=0,
    verbose=1,
)
save_best_model.set_weights(model_s2pop.get_weights())

zero_mask_train = y_train.sum(axis=(1, 2, 3)) != 0.0
y_train_reduced = y_train[zero_mask_train]
x_train_reduced = x_train[zero_mask_train]

zero_mask_val = y_val.sum(axis=(1, 2, 3)) != 0.0
y_val_reduced = y_val[zero_mask_val]
x_val_reduced = x_val[zero_mask_val]

x_train_disc = np.concatenate([
    model_s2pop.predict(x_train_reduced, batch_size=VAL_BS, verbose=0),
    y_train_reduced,
], axis=0)

y_train_disc = np.concatenate([
    np.ones(x_train_reduced.shape[0], dtype="float32"),
    np.zeros(y_train_reduced.shape[0], dtype="float32"),
], axis=0)

mask_disc = np.random.permutation(x_train_disc.shape[0])
x_train_disc = x_train_disc[mask_disc]
y_train_disc = y_train_disc[mask_disc]

x_val_disc = np.concatenate([
    model_s2pop.predict(x_val_reduced, batch_size=VAL_BS, verbose=0),
    y_val_reduced,
], axis=0)

y_val_disc = np.concatenate([
    np.ones(x_val_reduced.shape[0], dtype="float32"),
    np.zeros(y_val_reduced.shape[0], dtype="float32"),
], axis=0)

model_discriminator.optimizer.lr.assign(DISCRIMINATOR_PRERUN_LR)

model_discriminator.fit(
    x=x_train_disc,
    y=y_train_disc,
    validation_data=(x_val_disc, y_val_disc),
    shuffle=True,
    epochs=DISCRIMINATOR_PRERUN_FITS,
    initial_epoch=0,
    batch_size=DISCRIMINATOR_PRERUN_BS,
    validation_batch_size=VAL_BS,
    use_multiprocessing=True,
    workers=0,
    verbose=1,
)

# ------------------------------------ START LOOP ------------------------------------------- #
model_s2pop.compile(optimizer=OPTIMIZER_PRED, loss=test_loss, metrics=METRICS)
best_val_loss = model_s2pop.evaluate(x_val, y_val, verbose=0)[0]

for idx, fit in enumerate(fits):
    use_epoch = fit["epochs"]
    use_bs = fit["bs"]
    use_lr = fit["lr"]
    use_ie = fit["ie"]

    model_s2pop.optimizer.lr.assign(use_lr)

    model_s2pop.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        batch_size=use_bs,
        validation_batch_size=VAL_BS,
        use_multiprocessing=True,
        workers=0,
        verbose=1,
        callbacks=[
            SaveBestModel(),
            OverfitProtection(
                patience=3,
                difference=0.20, # 20% overfit allowed
                offset_start=3, # disregard overfit for the first epoch
            ),
        ],
    )

    model_s2pop.set_weights(save_best_model.get_weights())

    print("Test data:")
    model_s2pop.evaluate(x_test, y_test)

    if idx + 1 == len(fits):
        break

    x_train_disc = np.concatenate([
        model_s2pop.predict(x_train_reduced, batch_size=VAL_BS, verbose=0),
        y_train_reduced,
    ], axis=0)

    y_train_disc = np.concatenate([
        np.ones(x_train_reduced.shape[0], dtype="float32"),
        np.zeros(y_train_reduced.shape[0], dtype="float32"),
    ], axis=0)

    mask_disc = np.random.permutation(x_train_disc.shape[0])
    x_train_disc = x_train_disc[mask_disc]
    y_train_disc = y_train_disc[mask_disc]

    x_val_disc = np.concatenate([
        model_s2pop.predict(x_val, batch_size=VAL_BS, verbose=0),
        y_val,
    ], axis=0)

    y_val_disc = np.concatenate([
        np.ones(x_val.shape[0], dtype="float32"),
        np.zeros(y_val.shape[0], dtype="float32"),
    ], axis=0)

    model_discriminator.optimizer.lr.assign(use_lr)

    model_discriminator.fit(
        x=x_train_disc,
        y=y_train_disc,
        validation_data=(x_val_disc, y_val_disc),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        batch_size=use_bs,
        validation_batch_size=VAL_BS,
        use_multiprocessing=True,
        workers=0,
        verbose=1,
        callbacks=[
            OverfitProtection(
                patience=5,
                difference=0.25, # 20% overfit allowed
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
