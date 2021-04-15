import cv2
import tensorflow as tf
import numpy as np
import os
import models
import time
import re

test_loc=['./Data/DIV2K_valid_LR_unknown/X2','./Data/DIV2K_valid_LR_unknown 2/X3',
                './Data/DIV2K_valid_LR_unknown 3/X4','./Data/DIV2K_valid_LR_bicubic/X2',
                './Data/DIV2K_valid_LR_bicubic 2/X3','./Data/DIV2K_valid_LR_bicubic 3/X4']

real_path='./Data/DIV2K_valid_HR'


def Evaluation(model, network, test_path, real_path, scale):
    testdir = os.listdir(test_path)
    realdir = os.listdir(real_path)
    testdir.sort()
    realdir.sort()
    psnr_all = np.zeros(100)
    ssim_all = np.zeros(100)
    time_all = np.zeros(100)
    n = len(testdir)
    for i in range(n):
        img = cv2.imread(os.path.join(test_path, testdir[i]))
        if network=='SRCNN':
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))
        if network=='FSRCNN':
            img = cv2.resize(img, (img.shape[1], img.shape[0]))
        test_img= np.zeros((1, img.shape[0], img.shape[1], 3), dtype=np.double)
        test_img[0, :, :, :] = img

        start_time = time.time()
        pred = model.predict(test_img)
        end_time = time.time()
        running_time = end_time - start_time

        real = cv2.imread(os.path.join(real_path, realdir[i]))
        real_img = np.zeros((1, real.shape[0], real.shape[1], 3), dtype=np.double)
        real_img[0, :, :, :] = real

        psnr = tf.image.psnr(
            tf.clip_by_value(pred[0], 0, 255),
            tf.clip_by_value(real_img[0], 0, 255), max_val=255)
        ssim = tf.image.ssim(
             tf.cast(pred[0],tf.float32),
             tf.cast(real_img[0],tf.float32), max_val=255)
        psnr_all[i]=psnr
        ssim_all[i]=ssim
        time_all[i]=running_time
    return psnr_all, ssim_all, time_all

def test_model(network):
    if network=='SRCNN':
        weights_path='./weights/srcnn'
    if network=='FSRCNN':
        weights_path='./weights/fsrcnn'
    weightsdir=os.listdir(weights_path)
    weightsdir.sort()
    for file in weightsdir:
        weights=os.path.join(weights_path,file)
        scale=int(re.search("\d", file).group())
        index=weightsdir.index(file)
        test_path=test_loc[index]
        if network=='SRCNN':
            m=models.srcnn_predict()
            m.load_weights(weights)
        if network=='FSRCNN':
            m=models.FSRCNN_Predict(scale)
            m.load_weights(weights)
        psnr_all, ssim_all, time_all=Evaluation(m,network,test_path, real_path,scale)
        avg_psnr=np.mean(psnr_all)
        avg_ssim=np.mean(ssim_all)
        avg_time=np.mean(time_all)
        if 'un' in file:
            print(network+'-track2-scale{}'.format(scale))
            print('PSNR: %.4f' % avg_psnr)
            print('SSIM %.4f' % avg_ssim)
            print('running time: %.4f' % avg_time)
        else:
            print(network+'-track1-scale{}'.format(scale))
            print('PSNR: %.4f' % avg_psnr)
            print('SSIM %.4f' % avg_ssim)
            print('running time: %.4f' % avg_time)

def demo(network,scale):
    test_img = cv2.imread('./demo/input.png')
    if network=='SRCNN':
        weights_path='./weights/srcnn'
        weights = os.path.join(weights_path, 'SRCNN_x' + str(scale)+'.h5')
        m = models.srcnn_predict()
        m.load_weights(weights)

        test_img = cv2.resize(test_img, (test_img.shape[1] * scale, test_img.shape[0] * scale))
        temp = np.zeros((1, test_img.shape[0], test_img.shape[1], 3), dtype=np.double)
        temp[0, :, :, :] = test_img
        pred = m.predict(temp)
        cv2.imwrite('./demo/output_srcnn.png', pred[0])

    if network=='FSRCNN':
        weights_path='./weights/fsrcnn'
        weights = os.path.join(weights_path, 'FSRCNN_x' + str(scale) + '.h5')
        m = models.FSRCNN_Predict(scale)
        m.load_weights(weights)

        temp = np.zeros((1, test_img.shape[0], test_img.shape[1], 3), dtype=np.double)
        temp[0, :, :, :] = test_img

        pred = m.predict(temp)
        cv2.imwrite('./demo/output_fsrcnn.png', pred[0])



