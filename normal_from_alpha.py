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
import numpy as np
import time

start_time = time.time()
args = sys.argv[1:]
if not len(args):
    print('Please specify image file')
    sys.exit()

# Load image
impath = args[0]
dirname, filename = os.path.split(impath)
filename, ext = os.path.splitext(filename)
out_dist_impath = os.path.join(dirname, filename + '_dist' + ext)
out_nor_impath = os.path.join(dirname, filename + '_nor' + ext)
im = scipy.misc.imread(impath)
im = im.astype('int32')

# Get image size
shape = im.shape[:2]

black_pts = []
white_pts = []

# Init arrays
dist_array = np.zeros(shape)

# Build lists of coordinates where alpha is transparent(resp. opaque)
print('Building point lists...')
for y, row in enumerate(im[..., 3]):
    for x, val in enumerate(row):
        if val == 0:
            black_pts.append((x, y))
        else:
            white_pts.append((x, y))
print('Elapsed:', time.time() - start_time)

# Build 2d tree
print('Building kdtrees...')
kdt = ckdtree.cKDTree(black_pts)

# For each opaque point, get distance to nearest transp. point
# Store that in dist_array
for co in white_pts:
    dist = kdt.query(co, p=2)
    dist_array[co[1]][co[0]] = dist[0]

# Start again for points outside the shape
kdt = ckdtree.cKDTree(white_pts)
for co in black_pts:
    dist = kdt.query(co, p=2)
    dist_array[co[1]][co[0]] = -dist[0]+1
print('Elapsed:', time.time() - start_time)

# Get extrema
a_min, a_max = min(dist_array.ravel()), max(dist_array.ravel())
print(a_min, a_max)

# Normalize
dist_array = (dist_array - a_min) / (a_max - a_min) * np.pi / 2
dist_array = np.sin(dist_array)

a_min, a_max = min(dist_array.ravel()), max(dist_array.ravel())
print(a_min, a_max)

# Blur
dist_array = gaussian_filter(dist_array, 5.0)

# Calculate sobel filter on x and y axes
print('Calculating Sobel filter...')
nor_array = np.array((sobel(dist_array, axis=1), sobel(dist_array, axis=-0), dist_array))
nor_array = nor_array.swapaxes(0,-1)
nor_array = nor_array.swapaxes(0,1)
print('Elapsed:', time.time() - start_time)

# Invert x axis
x = nor_array[..., 0]
x *= -1

# Normalize
rg = nor_array[..., :-1]
rg[:] -= min(rg.ravel())
rg[:] /= max(rg.ravel())
rg[:] *= 2**8-1
nor_array[...,2] *= 2**8-1

nor_array = nor_array.astype('uint')

# Save
scipy.misc.imsave(out_dist_impath, dist_array)
scipy.misc.imsave(out_nor_impath, nor_array)

print('DONE')
