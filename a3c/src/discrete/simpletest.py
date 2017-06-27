from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from load_images import load_images_array
import os

input_img = Input(shape=(80, 160, 3))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name="conv_1")(input_img)
print x
x = MaxPooling2D((2, 2), border_mode='same', name="max_pool_1")(x)
print x
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name="conv_2")(x)
print x
x = MaxPooling2D((2, 2), border_mode='same', name="max_pool_2")(x)
print x
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name="conv_3")(x)
print x
x = MaxPooling2D((2, 2), border_mode='same', name="max_pool_3")(x)
print x
x = Convolution2D(4, 3, 3, activation='relu', border_mode='same', name="conv_4")(x)
print x
encoded = MaxPooling2D((2, 2), border_mode='same', name="max_pool_4")(x)

print encoded

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(4, 3, 3, activation='relu', border_mode='same', name="deconv_1")(encoded)
x = UpSampling2D((2, 2), name="up_samp_1")(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name="deconv_2")(x)
x = UpSampling2D((2, 2), name="up_samp_2")(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name="deconv_3")(x)
x = UpSampling2D((2, 2), name="up_samp_3")(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name="deconv_4")(x)
x = UpSampling2D((2, 2), name="up_samp_4")(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', name="deconv_5")(x)

print decoded

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

if os.path.isfile("autoencoder.h5"):
    autoencoder.load_weights("autoencoder.h5", by_name=True)
    print("Loaded model from disk")

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

#if os.path.isfile("X_train.npy") and os.path.isfile("X_test.npy"):
#    x_train = np.load("X_train.npy")
#    x_test = np.load("X_test.npy")
#else:
#    (x_train, x_test) = load_images_array()

(x_train, x_test) = load_images_array()

x_train /= 255.0
x_test /= 255.0

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print x_train.shape
print x_test.shape

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),
                           early_stopping])
 
# serialize model to JSON
autoencoder_json = autoencoder.to_json()
with open("autoencoder.json", "w") as json_file:
    json_file.write(autoencoder_json)
# serialize weights to HDF5
autoencoder.save_weights("autoencoder.h5")
print("Saved weights to disk")
               
# encode and decode some digits
# note that we take them from the *test* set
decoded_imgs = autoencoder.predict(x_test) #* 255.0

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
