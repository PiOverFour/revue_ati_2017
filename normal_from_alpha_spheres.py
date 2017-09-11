#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Peinture normale : Approche picturale du rendu de surfaces
# Copyright (C) 2017  Damien Picard dam.pic AT free.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import scipy
from scipy.spatial import ckdtree
from scipy.ndimage.filters import sobel, gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import time

start_time = time.time()
args = sys.argv[1:]
if not len(args):
    print('Please specify image file')
    sys.exit()

# load image
impath = args[0]
dirname, filename = os.path.split(impath)
filename, ext = os.path.splitext(filename)
out_dist_impath = os.path.join(dirname, filename + '_dist' + ext)
out_nor_impath = os.path.join(dirname, filename + '_nor' + ext)
im = scipy.misc.imread(impath)
im = im.astype('int32')

# Get image size
shape = im.shape[:2]

# Threshold for half transparency
thresh_mask = im[..., 3] < 30
im[thresh_mask, 3] = 0.0


print("Calculating distance transform...")
dist_array = distance_transform_edt(im[..., 3])
print('Elapsed:', time.time() - start_time)


def make_mask(shape, radius, center=(0, 0)):
    '''from http://stackoverflow.com/a/28175622/4561348'''
    c, r = shape
    center = (center[0]/2, center[1]/2)
    y, x = np.ogrid[-center[0]:c-center[0], -center[1]:r-center[1]]
    R_squared = (center[0] - x)**2 + (center[1] - y)**2
    distance = np.sqrt(R_squared)
    # Weigh it inversely to distance from center
    weight = radius - distance, 0 # 1 - distance/radius
    # Spherize
    weight = np.sqrt(np.maximum(radius*radius - distance*distance, 0))
    return weight


# Pad array to avoid slicing outside it
max_dist = int(np.ceil((np.max(dist_array))))
padded_shape = (shape[0] + 2*max_dist, shape[1] + 2*max_dist)
print(padded_shape)
height_array = np.zeros(padded_shape)

print('Calculating spheres...')

circle_lut = {}

for (x, y), dist_pt in np.ndenumerate(dist_array):
    if dist_pt <= 0:  # outside shape
        continue

    center = [x, y]
    dist_to_border = dist_pt

    # build circle array
    ceil_dist = np.ceil(dist_to_border)
    if ceil_dist in circle_lut:
        circle_array = circle_lut[ceil_dist]
    else:
        circle_array = make_mask((ceil_dist*2,)*2, dist_to_border, (dist_to_border,)*2)
        circle_lut[ceil_dist] = circle_array

    offset = int(ceil_dist // 2 > 0.5)

    x_min = int(center[0] - ceil_dist + max_dist + offset)
    x_max = int(center[0] + ceil_dist + max_dist + offset)
    y_min = int(center[1] - ceil_dist + max_dist + offset)
    y_max = int(center[1] + ceil_dist + max_dist + offset)

    height_array[x_min:x_max, y_min:y_max] = np.maximum(height_array[x_min:x_max, y_min:y_max], circle_array)

# Remove padding
height_array = height_array[max_dist:shape[0]+max_dist, max_dist:shape[1]+max_dist]

height_array *= height_array
print('Elapsed:', time.time() - start_time)

# Get extrema
a_min, a_max = np.min(height_array), np.max(height_array)
#print(a_min, a_max)

# Normalize
height_array = (height_array - a_min) / (a_max - a_min) * np.pi / 2

a_min, a_max = np.min(height_array), np.max(height_array)


# Calculate sobel filter on x and y axes
nor_array = np.array((sobel(height_array, axis=1), sobel(height_array, axis=-0), height_array))
nor_array = nor_array.swapaxes(0,-1)
nor_array = nor_array.swapaxes(0,1)

# Invert x axis
x = nor_array[..., 0]
x *= -1

# Normalize
rg = nor_array[..., :-1]
rg[:] -= np.min(rg)
rg[:] /= np.max(rg)
rg[:] *= 2**8-1
nor_array[...,2] = 2**8-1

nor_array = nor_array.astype('uint')

# Save
scipy.misc.imsave(out_dist_impath, height_array)
scipy.misc.imsave(out_nor_impath, nor_array)

print('DONE')
