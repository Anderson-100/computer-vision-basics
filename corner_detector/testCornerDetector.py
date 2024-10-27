import numpy as np
import matplotlib.pyplot as plt
import utils
from skimage import data
from detectCorners import detectCorners

# I = data.checkerboard()
# I = utils.imread('../data/polymer-science-umass.jpg')
I = utils.imread('../data/tissue.jpeg')

plt.figure(1)
cx, cy, cs = detectCorners(I, True, 1.5, 0.005)
plt.subplot(121)
#from IPython import embed; embed(); exit(-1)
if I.ndim == 2:
    plt.imshow(I, cmap='gray')
else:
    plt.imshow(I)
plt.plot(cx, cy, 'r.')
plt.title('Simple Corners')
plt.axis('off')

cx, cy, cs = detectCorners(I, False, 1.5, 0.0001)
plt.subplot(122)
if I.ndim == 2:
    plt.imshow(I, cmap='gray')
else:
    plt.imshow(I)
plt.plot(cx, cy, 'g.')
plt.title('Harris Corners')
plt.axis('off')

plt.show()


