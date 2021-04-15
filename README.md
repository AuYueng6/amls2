## Introduction

This project solves the challenge of single image super-resolution of both track 1 and track 2 in NTIRE 2017 by implementing SRCNN and FSRCNN models.
Dataset used in this project is availble on https://data.vision.ee.ethz.ch/cvl/DIV2K/

A demo image is provided in demo folder. By running the demo function in main.py, you can use the two models in this project to generate SR images. You can also use your own image, but the image format should be PNG and named input.png and placed in the demo folder.
## Files
* models.py: models for both training and test
* train.py: training process of the models
* test.py: includes test process and a demo function

## Dependencies
* Python 3.6
* Tensorflow = 2.3.1
* numpy
* time
* re
* cv2 >= 3.xx
