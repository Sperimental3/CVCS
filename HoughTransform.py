# Code for the final spatial extent of the tensor shaping
# Hough Transform and RoI align application

import numpy as np
import cv2
import torch
import os
from torchvision.ops import roi_align
import matplotlib.pyplot as plt

# inPath = "C:\\Users\\Matteo\\Desktop\\Coins_Dataset"
# a more generic version of the inPath
inPath = "./Coins_Dataset/"

# array for circles radii analysis
# radii_analysis = np.empty((113, 3))

# by performing some analysis on the position and the radii of the circles we discover that
# the maximum radius is 144 and the minimum is 24, we want to have just the coin image for feeding
# the CNN and we also need a mini-batch of images, so they have to be of the same shape(possibly
# reduced, for computational reason), to achieve both of these goals we use roi_align

# preparing the tensor for a batch of 113 images each with 3 channels(RGB), and a 100x100 dimension(HxW)
# this dimension is due to the fact that the smallest coin has a diameter of 48 pixels, we  want to add
# a padding of 26 pixels for arriving to 100 and for the non-perfect localization of the Hough transform
images_batch = torch.empty((113, 3, 100, 100))


def hough_transform():
    for imagePath, i in zip(os.listdir(inPath), range(113)):
        # image reading and preparation
        im = cv2.imread(f"{inPath}\\{imagePath}")

        # HT needs a grey scale image
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im = torch.from_numpy(np.moveaxis(im, 2, 0)).unsqueeze_(0).type(torch.float32)

        # the minDist is 662 because we don't want more than one circle, so we calculate it by considering
        # the maximum dimension of both the height(547) and the width(373) of our images mini-batch,
        # formula:(H^2 + W^2)^(1/2)
        circle = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=662,
                                  param1=300, param2=20, minRadius=20, maxRadius=150)

        circle = (np.around(circle)).astype(np.int32).squeeze()

        # calculating the coordinates of the bounding box as described before,
        # performing then a clipping to make sure of not overcome the borders of the image
        x1 = np.clip((circle[0] - circle[2] - 26), a_min=0, a_max=None)
        y1 = np.clip((circle[1] - circle[2] - 26), a_min=0, a_max=None)
        x2 = np.clip((circle[0] + circle[2] + 26), a_min=None, a_max=im.shape[-1])
        y2 = np.clip((circle[1] + circle[2] + 26), a_min=None, a_max=im.shape[-2])

        # preparing the single box with a dimension equal to that of the image
        box_shape = torch.tensor((0, x1, y1, x2, y2), dtype=torch.float32).unsqueeze_(0)

        # actual population of the mini-batch with the roi_align
        images_batch[i] = roi_align(im, box_shape, (100, 100), sampling_ratio=2)

        # rows for analysis
        '''
        circles = (np.around(circles)).astype(np.int32)
        # print(f"for index {i}: ", circles, circles.shape, type(circles))
        # radii_analysis[i, :] = circles
        '''

        # if we want each image to have the detection circle drawn uncomment what follow
        '''
        if circle is not None:
            circle = (np.around(circle)).astype(np.int32)
            
            for j in circle[0, :]:
                center = (j[0], j[1])
                # circle center
                cv2.circle(im, center, 1, (0, 255, 0), 3)
                # circle outline
                radius = j[2]
                cv2.circle(im, center, radius, (0, 255, 0), 3)
        
        # plotting the images to visually verify that the circles were corrects
        if i in range(5):
            plt.imshow(im)
            plt.show()
        '''
    return images_batch

    # row for analysis
    # print(radii_analysis.max(axis=0), radii_analysis.min(axis=0))
    # print(images_batch.type(), images_batch.shape)


# code to print some results as examples
def plot_examples():
    for image, i in zip(images_batch, range(113)):
        print(f"The image with index {i} is:")
        print(image.shape)

        if i in range(5):
            im = torch.moveaxis(image, 0, 2).type(torch.int32)

            plt.imshow(im)
            plt.show()


