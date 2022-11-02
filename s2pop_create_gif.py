import os
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import buteo as beo
from glob import glob
from functools import cmp_to_key


FOLDER = "C:/Users/caspe/Desktop/unicef_code/gifs/*.tif"
DIVISIONS = 1

def cmp_items(a, b):
    a_int = int(os.path.splitext(os.path.basename(a))[0].split("_")[4])
    b_int = int(os.path.splitext(os.path.basename(b))[0].split("_")[4])
    if a_int > b_int:
        return 1
    elif a_int == b_int:
        return 0
    else:
        return -1

images = glob(FOLDER)
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
        interpolated = np.linspace(arrays[idx-1], arr, DIVISIONS, axis=0)
        for i in range(DIVISIONS):
            arrays_interpolated.append(interpolated[i, :, :])

fig, ax = plt.subplots()

im = plt.imshow(arrays_interpolated[0], cmap=plt.get_cmap("viridis"), vmin=0.0, vmax=1.0, interpolation="nearest")
times = np.arange(0.0, len(arrays), len(arrays) / len(arrays_interpolated), dtype="float32")
label = ax.text(4, 4, f"{round(times[0])}", ha='left', va='top', fontsize=15, color="#BBBBBB")


def updatefig(j):
    im.set_array(arrays_interpolated[j])
    label.set_text(f"{round(times[j])}")

    return [im, label]

# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(len(arrays_interpolated)), interval=10, blit=True)

fig.patch.set_facecolor('#22252A')
plt.axis('off')
plt.tight_layout()
plt.show()