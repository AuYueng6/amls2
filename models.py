from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,PReLU

def srcnn_train():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(64 ,(9 ,9) ,activation='relu' ,padding='valid' ,input_shape=(144 ,144 ,3)))
    SRCNN.add(Conv2D(32 ,(3 ,3) ,activation='relu' ,padding='valid'))
    SRCNN.add(Conv2D(32, (1, 1), activation='relu', padding='valid'))
    SRCNN.add(Conv2D(3 ,(5 ,5) ,padding='valid'))
    return SRCNN

def srcnn_predict():
    SRCNN_p = Sequential()
    SRCNN_p.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 3)))
    SRCNN_p.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    SRCNN_p.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    SRCNN_p.add(Conv2D(3, (5, 5), padding='same'))
    return SRCNN_p


def FSRCNN(scale):
    shape = int(48 / scale)

    fsrcnn = Sequential()

    fsrcnn.add(Conv2D(128, (5, 5), padding='same', name='conv1', input_shape=(shape, shape, 3)))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu1'))
    fsrcnn.add(Conv2D(32, (1, 1), padding='same', name='conv2'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu2'))
    fsrcnn.add(Conv2D(32, (3, 3), padding='same', name='conv3'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu3'))
    fsrcnn.add(Conv2D(32, (3, 3), padding='same', name='conv4'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu4'))
    fsrcnn.add(Conv2D(32, (3, 3), padding='same', name='conv5'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu5'))
    fsrcnn.add(Conv2D(128, (1, 1), padding='same', name='conv6'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu6'))
    fsrcnn.add(Conv2DTranspose(3, (9, 9), padding='same', strides=(scale, scale), name=str(scale)))
    return fsrcnn


def FSRCNN_Predict(scale):
    fsrcnn = Sequential()

    fsrcnn.add(Conv2D(128, (5, 5), padding='same', name='conv1', input_shape=(None, None, 3)))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu1'))
    fsrcnn.add(Conv2D(32, (1, 1), padding='same', name='conv2'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu2'))
    fsrcnn.add(Conv2D(32, (3, 3), padding='same', name='conv3'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu3'))
    fsrcnn.add(Conv2D(32, (3, 3), padding='same', name='conv4'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu4'))
    fsrcnn.add(Conv2D(32, (3, 3), padding='same', name='conv5'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu5'))
    fsrcnn.add(Conv2D(128, (1, 1), padding='same', name='conv6'))
    fsrcnn.add(PReLU(shared_axes=[1, 2], name='prelu6'))
    fsrcnn.add(Conv2DTranspose(3, (9, 9), padding='same', strides=(scale, scale), name=str(scale)))

    return fsrcnn