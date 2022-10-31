import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import buteo as beo
from glob import glob
from functools import cmp_to_key



def cmp_items(a, b):
    a_int = int(os.path.splitext(os.path.basename(a))[0].split("_")[4])
    b_int = int(os.path.splitext(os.path.basename(b))[0].split("_")[4])
    if a_int > b_int:
        return 1
    elif a_int == b_int:
        return 0
    else:
        return -1

images = glob("/home/casper/Desktop/UNICEF/gifs/*.tif")
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
        interpolated = np.linspace(arrays[idx-1], arr, 10, axis=0)
        for i in range(10):
            arrays_interpolated.append(interpolated[i, :, :])



fig = plt.figure() # make figure

# make axesimage object
# the vmin and vmax here are very important to get the color map correct
im = plt.imshow(arrays_interpolated[0], cmap=plt.get_cmap("viridis"), vmin=0.0, vmax=1.0, interpolation="nearest")

# function to update figure
def updatefig(j):
    # set the data in the axesimage object
    im.set_array(arrays_interpolated[j])
    # return the artists set
    return [im]

# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(len(arrays_interpolated)), interval=50, blit=True)

plt.show()