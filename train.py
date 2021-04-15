import models
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import re

train_data_dir=['./Data/DIV2K_train_HR','./Data/DIV2K_train_LR_bicubic/X2','./Data/DIV2K_train_LR_bicubic 2/X3',
                './Data/DIV2K_train_LR_bicubic 3/X4','./Data/DIV2K_train_LR_unknown/X2','./Data/DIV2K_train_LR_unknown 2/X3',
                './Data/DIV2K_train_LR_unknown 3/X4']


def PSNR(y_true, y_pred):
    psnr = tf.image.psnr(
                tf.clip_by_value(y_pred,0,255),
                tf.clip_by_value(y_true,0,255), max_val=255)
    return psnr

def load_srcnn_data(path, rsize,scale):
    data = []
    listdir = os.listdir(path)
    listdir.sort()
    for file in listdir:
        img = cv2.imread(os.path.join(path, file))
        if rsize == 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))
        w_number = int((img.shape[0] - 144) / 600) #small images cropped from the original image
        h_number = int((img.shape[1] - 144) / 600)
        for i in range(w_number):
            for j in range(h_number):
                img_patch = img[i * 600:i * 600 + 144, j * 600:j * 600 +144 , :]
                lr = np.zeros((144, 144, 3), dtype=np.double)
                hr = np.zeros((130, 130, 3), dtype=np.double) #the  output  image  of  the  model  would  lose  7pixels of the boundary
                if rsize == 0:
                    hr = img_patch[7:-7, 7:-7, :]
                    data.append(hr)
                else:
                    lr = img_patch

                    data.append(lr)
    data = np.array(data)
    return data


def load_fsrcnn_data(path, lr, scale):
    data = []
    listdir = os.listdir(path)
    listdir.sort()
    hsize = 48
    hstep = 96
    lsize = int(hsize / scale)
    lstep = int(hstep / scale)
    for file in listdir:
        img = cv2.imread(os.path.join(path, file))
        if lr == 1:
            w_number = int((img.shape[0] - lsize) / lstep) #small images cropped from original image
            h_number = int((img.shape[1] - lsize) / lstep)
            for i in range(w_number):
                for j in range(h_number):
                    img_patch = img[i * lstep:i * lstep + lsize, j * lstep:j * lstep + lsize, :]
                    data.append(img_patch)
        else:
            w_number = int((img.shape[0] - hsize) / hstep)
            h_number = int((img.shape[1] - hsize) / hstep)
            for i in range(w_number):
                for j in range(h_number):
                    img_patch = img[i * hstep:i * hstep + hsize, j * hstep:j * hstep + hsize, :]
                    data.append(img_patch)

    data = np.array(data)
    return data

def train_srcnn():
    weightsSaveDir = "./weights/srcnn/"
    if not os.path.isdir(weightsSaveDir):
        os.makedirs(weightsSaveDir)

    adam = Adam(lr=0.0001)

    for i in range(2,8):
        SRCNN=models.srcnn_train()
        SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=[PSNR])

        if i <5:
            k=i
            lr_img = load_srcnn_data(train_data_dir[i - 1], 1, k)
            NAME='SRCNN_x'+str(k)
            chkpt = os.path.join(weightsSaveDir, NAME +'.h5')
            cp_callback=[tf.keras.callbacks.ModelCheckpoint(chkpt, verbose=1,monitor='val_loss', mode='min',
                                                             save_weights_only=True, save_best_only=True)]
        else:
            k=i-3
            lr_img = load_srcnn_data(train_data_dir[i - 1], 1, k)
            NAME='SRCNN_un_x'+str(k)
            chkpt = os.path.join(weightsSaveDir, NAME +'.h5')
            cp_callback = [tf.keras.callbacks.ModelCheckpoint(chkpt, verbose=1,monitor='val_loss', mode='min',
                                                             save_weights_only=True, save_best_only=True)]




        SRCNN.fit(lr_img,hr_img,
                  batch_size=16,
                  epochs=50,validation_split=0.2,
                 callbacks=cp_callback)

def train_fsrcnn():
    weightsSaveDir = "./weights/fsrcnn/"
    if not os.path.isdir(weightsSaveDir):
        os.makedirs(weightsSaveDir)

    adam = Adam(lr=0.0001)

    for i in range(1,7):
        scale = int(re.search("\d$", train_data_dir[i]).group())
        lr_img=load_fsrcnn_data(train_data_dir[i],1,scale)

        fsrcnn = models.FSRCNN(scale)
        fsrcnn.compile(optimizer=adam, loss='mse', metrics=[PSNR])

        if i < 4:
            NAME='FSRCNN_x' + str(scale)
            chkpt = os.path.join(weightsSaveDir, NAME + '.h5' )
            cp_callback = [tf.keras.callbacks.ModelCheckpoint(chkpt, verbose=1, monitor='loss', mode='min',
                                                             save_weights_only=True, save_best_only=True)]
        else:
            NAME='FSRCNN_un_x' + str(scale)
            chkpt = os.path.join(weightsSaveDir, NAME + '.h5')
            cp_callback = [tf.keras.callbacks.ModelCheckpoint(chkpt, verbose=1, monitor='loss', mode='min',
                                                             save_weights_only=True, save_best_only=True)]



        fsrcnn.fit(lr_img, hr_img_f,
                   batch_size=128,
                   epochs=50,
                   validation_split=0.2,
                   callbacks=cp_callback)


if __name__ == '__main__':
    hr_img = load_srcnn_data(train_data_dir[0], 0, 1)
    hr_img_f = load_fsrcnn_data(train_data_dir[0], 0, 4)
    train_srcnn()
    train_fsrcnn()