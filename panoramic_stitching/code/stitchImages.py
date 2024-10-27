import utils
import numpy as np
from matplotlib import pyplot as plt

from detectCorners import detectCorners
from extractFeatures import extractFeatures
from computeMatches import computeMatches
from mergeImages import mergeImages
from ransac import ransac

# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 4
#
# Read images

im1 = utils.imread('../data/umass_building_left.jpg') #left image

i = 5
im2 = utils.imread(f'../data/umass_building_right{i}.jpg') #right image (change this to right2, right3, ...)

sigma = 1.5
threshold = 0.0005
max_corners = 200
isSimple = False
c1 = detectCorners(im1, isSimple, sigma, threshold)
c2 = detectCorners(im2, isSimple, sigma, threshold)

n1 = min(max_corners, c1.shape[1])
c1 = c1[:, 0:n1]
n2 = min(max_corners, c2.shape[1])
c2 = c2[:, 0:n2]

#Compute feature descriptors
patch_radius = 5
f1 = extractFeatures(im1, c1, patch_radius)
f2 = extractFeatures(im2, c2, patch_radius)

#Compute matches
matches = computeMatches(f1, f2)
utils.showMatches(im1, im2, c1, c2, matches, title='All correspondences')
plt.tight_layout()
plt.show()

#Estimate transformation
inliers, transf = ransac(matches, c1, c2, max_iter=1100)
good_matches = np.zeros_like(matches)-1
good_matches[inliers] = matches[inliers]
utils.showMatches(im1, im2, c1, c2, good_matches, title='Inliers')
plt.tight_layout()
plt.show()

#Warp images
stitch_im = mergeImages(im1, im2, transf)
plt.figure(figsize=(20,8))
plt.imshow(stitch_im)
plt.title('Stitched Image', fontsize=22)
plt.tight_layout()
plt.show()

print(f"Image {i}:")
print(transf)
print("# inliers:", len(inliers))