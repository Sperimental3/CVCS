# code for image analysis, to understand the dimensions and how to pre-process the images
# before feeding them to the network

import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from torchvision.ops import roi_align

inPath = "C:\\Users\\Matteo\\Desktop\\Coins_Dataset"

# array for analyzing images dimensions
dimensions = np.empty((113, 3))

for imagePath, i in zip(os.listdir(inPath), range(113)):
    ex = cv2.imread(f"{inPath}\\{imagePath}")
    dimensions[i] = ex.shape

print(np.unique(dimensions[:, 0]))

print(dimensions)
print(dimensions.max(axis=0), dimensions.min(axis=0), dimensions.mean(axis=0))

ex1 = "C:\\Users\\Matteo\\Desktop\\Coins_Dataset\\s-34882-8947.jpg"

im = cv2.imread(ex1)

img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

print(type(img), img.shape)

plt.imshow(img)
plt.show()

# a first try whit roi_align

img = torch.from_numpy(np.moveaxis(img, 2, 0)).unsqueeze_(0).type(torch.float32)

print(img.type(), img.shape)

box_shape = torch.tensor((0, 0, 0, img.shape[-1], img.shape[-2]), dtype=torch.float32).unsqueeze_(0)

print(box_shape.type(), box_shape.shape)
output = roi_align(img, box_shape, (190, 190), sampling_ratio=2)

output = torch.moveaxis(torch.squeeze(output), 0, 2).type(torch.int32)

print(output.type(), output.shape)
print(output)

plt.imshow(output)
plt.show()
