import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import buteo as beo
from glob import glob
from functools import cmp_to_key


def cmp_items(a, b):
    idx_location = 5
    a_int = int(os.path.splitext(os.path.basename(a))[0].split("_")[idx_location])
    b_int = int(os.path.splitext(os.path.basename(b))[0].split("_")[idx_location])
    if a_int > b_int:
        return 1
    elif a_int == b_int:
        return 0
    else:
        return -1

def ready_image(glob_path, divisions=5):
    images = glob(glob_path)
    images.sort(key=cmp_to_key(cmp_items))

    arrays = []
    for img in images:
        img = beo.raster_to_array(img)[:, :, 0]
        arrays.append(img)

    arrays_interpolated = []
    for idx, arr in enumerate(arrays):
        if idx == 0:
            arrays_interpolated.append(arr)
        else:
            interpolated = np.linspace(arrays[idx-1], arr, divisions, axis=0)
            for i in range(divisions):
                arrays_interpolated.append(interpolated[i, :, :])

    return arrays_interpolated

DIVISIONS = 10

rows = 1
cols = 1
fig, ax = plt.subplots(nrows=rows, ncols=cols)
# ax1, ax2 = ax
ax1 = ax

img1 = ready_image("/home/casper/Desktop/UNICEF/gifs/s2pop_gan_v17_north-america_30_*.tif", divisions=DIVISIONS)
img2 = ready_image("/home/casper/Desktop/UNICEF/gifs/s2pop_gan_v17_north-america_90_*.tif", divisions=DIVISIONS)

im1 = ax1.imshow(img1[0], cmap=plt.get_cmap("viridis"), vmin=0.0, vmax=1.0, interpolation="nearest"); ax1.axis("off"); ax1.set_title('San Diego', color="#BBBBBB")
# im2 = ax2.imshow(img2[0], cmap=plt.get_cmap("viridis"), vmin=0.0, vmax=1.0, interpolation="nearest"); ax2.axis("off"); ax2.set_title('Palm Springs', color="#BBBBBB")

times = np.arange(0.0, len(img1), len(img1) / len(img1), dtype="float32")

def updatefig(j):
    im1.set_array(img1[j])
    # im2.set_array(img2[j])

    return [
        im1,
        # im2,
    ]

DPI = 96
HEIGHT = 1000
WIDTH = 1000
plt.figure(figsize=(HEIGHT / DPI, WIDTH / DPI), dpi=DPI)
plt.tight_layout()
fig.patch.set_facecolor('#22252A')
anim = animation.FuncAnimation(fig, updatefig, frames=range(len(img1)), interval=50, blit=True)


anim.save(r"/home/casper/Desktop/UNICEF/animation_02.mp4", writer=animation.FFMpegWriter(fps=25, bitrate=25000))

# plt.show()