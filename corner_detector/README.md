# Corner Detector

A corner detector is a simple and fairly effective way of detecting objects within an image.

Here, we implement a simple corner detector and a Harris corner detector. 

A simple corner detector looks for large changes in pixel values in 8 directions: up, down, left, right, and the four directions between them.

A Harris corner detector uses the same concept but applies it to all possible directions using the magic of linear algebra. https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
