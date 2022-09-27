from glob import glob
import numpy as np
import tensorflow as tf

from buteo.raster.patches import get_patches
from buteo.utils.core_utils import progress
from s2super.super_sample import get_model, super_sample

model_s2super = get_model()

def get_training_data_save(
    folder,
    out_folder,
    image_to_sample=10,
    masked_vals=[0, 1, 8, 9],
    masked_frac=0.2,
    tile_size=64,
    normalise=True,
    normalise_value=10000.0,
    offsets=3,
    limit=False,
    limit_per_place=100,
):
    images = glob(folder + "*.npz")
    shuffle_idx = np.random.permutation(len(images))

    processing_round = 0
    processed = 0
    while processed < len(images):
        progress(processing_round, len(images) // image_to_sample, "Generating")
        chosen = []

        for idx in range(processed, processed + image_to_sample):
            img_idx = shuffle_idx[idx]
            chosen.append(images[img_idx])

        merge_bands = []
        # merge_nir_lr = []
        # merge_scl = []
        merge_sincos = []

        for image in chosen:
            loaded = np.load(image)

            bands = loaded["bands"]
            # nir_lr = loaded["nir"]
            scl = loaded["scl"]
            sincos = loaded["sincos"]

            super_bands = super_sample(bands, fit_data=False, verbose=False, preloaded_model=model_s2super)

            bands = super_bands

            overlap_bands, _, _ = get_patches(bands, tile_size=tile_size, number_of_offsets=offsets)
            # overlap_nir_lr, _, _ = get_patches(nir_lr, tile_size=tile_size, number_of_offsets=offsets)
            overlap_scl, _, _ = get_patches(scl, tile_size=tile_size, number_of_offsets=offsets)
            overlap_sincos, _, _ = get_patches(sincos, tile_size=tile_size, number_of_offsets=offsets)
            
            overlap_bands = np.concatenate(overlap_bands, axis=0)
            # overlap_nir_lr = np.concatenate(overlap_nir_lr, axis=0)
            overlap_scl = np.concatenate(overlap_scl, axis=0)
            overlap_sincos = np.concatenate(overlap_sincos, axis=0)

            limit = int(tile_size * tile_size * masked_frac)
            mask = np.isin(overlap_scl, np.array(masked_vals, dtype="uint16")).sum(axis=(1, 2))[:, 0] < limit
            shuffle_mask = np.random.permutation(mask.sum())

            if normalise:
                overlap_bands = (overlap_bands[mask] / normalise_value).astype("float32")
                # overlap_nir_lr = overlap_nir_lr[mask] / normalise_value
                overlap_scl = overlap_scl[mask]
                overlap_sincos = overlap_sincos[mask]
            else:
                overlap_bands = overlap_bands[mask]
                # overlap_nir_lr = overlap_nir_lr[mask]
                overlap_scl = overlap_scl[mask]
                overlap_sincos = overlap_sincos[mask]
            
            overlap_bands = overlap_bands[shuffle_mask]
            # overlap_nir_lr = overlap_nir_lr[shuffle_mask].astype("float32")
            # overlap_scl = overlap_scl[shuffle_mask].astype("uint16")
            overlap_sincos = overlap_sincos[shuffle_mask].astype("float32")
        
            merge_bands.append(overlap_bands)
            # merge_nir_lr.append(overlap_nir_lr)
            # merge_scl.append(overlap_scl)
            merge_sincos.append(overlap_sincos)

        bands = np.concatenate(merge_bands, axis=0)
        # nir_lr = np.concatenate(merge_nir_lr, axis=0)
        # scl = np.concatenate(merge_scl, axis=0)
        sincos = np.concatenate(merge_sincos, axis=0)

        shuffle_mask = np.random.permutation(bands.shape[0])
        bands = bands[shuffle_mask]
        # nir_lr = nir_lr[shuffle_mask]
        # scl = scl[shuffle_mask]
        sincos = sincos[shuffle_mask].mean(axis=(1, 2))

        processed += image_to_sample
        processing_round += 1

        progress(processing_round, len(images) // image_to_sample, "Generating")

        # rgb = bands[:, :, :, [0, 1, 2]]
        # nir = bands[:, :, :, 3][:, :, :, np.newaxis]

        if limit:
            bands = bands[:int(limit_per_place * image_to_sample)]
            sincos = sincos[:int(limit_per_place * image_to_sample)]
            # rgb = rgb[:int(limit_per_place * image_to_sample)]
            # nir = nir[:int(limit_per_place * image_to_sample)]
            # nir_lr = nir_lr[:int(limit_per_place * image_to_sample)]

        samples = bands.shape[0]

        np.savez_compressed(
            f"{out_folder}{processing_round}_train_{samples}",
            bands=bands,
            sincos=sincos,
            # nir_lr=nir_lr,
            # rgb=rgb,
            # nir=nir,
        )


DATA_FOLDER = "F:/SenGlobal_data/"
OUT_FOLDER = "D:/CFI/prep_data/"
PLACES = 30
NORMALISE = 10000.0
TILE_SIZE = 64
OFFSETS = 0
MASKED_FRAC = 0.2
LIMIT = True
LIMIT_PER_PLACE = 150

get_training_data_save(
    DATA_FOLDER,
    OUT_FOLDER,
    image_to_sample=PLACES,
    masked_frac=MASKED_FRAC,
    normalise=False,
    normalise_value=NORMALISE,
    offsets=OFFSETS,
    limit=LIMIT,
    limit_per_place=LIMIT_PER_PLACE,
)

# merged_rgb = []
# merged_nir = []
# merged_nir_lr = []
merged_bands = []
merged_sincos = []
images = glob(OUT_FOLDER + "*.npz")
for idx, img in enumerate(images):
    loaded = np.load(img)
    merged_bands.append(loaded["bands"])
    merged_sincos.append(loaded["sincos"])
    # merged_rgb.append(loaded["rgb"])
    # merged_nir.append(loaded["nir"])
    # merged_nir_lr.append(loaded["nir_lr"])
    progress(idx + 1, len(images), "Loaded")

print("Concatenating")
merged_bands = np.concatenate(merged_bands, axis=0)
merged_sincos = np.concatenate(merged_sincos, axis=0)
# merged_rgb = np.concatenate(merged_rgb, axis=0)
# merged_nir = np.concatenate(merged_nir, axis=0)
# merged_nir_lr = np.concatenate(merged_nir_lr, axis=0)

print("Saving")
shuffle_mask = np.random.permutation(merged_bands.shape[0])
# shuffle_mask = np.random.permutation(merged_rgb.shape[0])
np.savez_compressed(
    f"{OUT_FOLDER}train_global",
    bands=merged_bands[shuffle_mask],
    sincps=merged_sincos[shuffle_mask],
    # rgb=merged_rgb[shuffle_mask],
    # nir=merged_nir[shuffle_mask],
    # nir_lr=merged_nir_lr[shuffle_mask],
)
