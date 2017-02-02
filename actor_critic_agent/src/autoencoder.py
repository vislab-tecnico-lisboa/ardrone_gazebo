from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model, model_from_json
import os

class autoencoder_network(object):
    def __init__(self):
        self.autoencoder = None
        self.encoder = None
        
        self.create_load_network()
        
    def run_network(self, input_image):
        features = self.encoder.predict(input_image)
        return features

    def create_load_network(self):
        # load json and create model
        if os.path.isfile("autoencoder.json"):
          json_file = open('autoencoder.json', 'r')
          autoencoder_json = json_file.read()
          json_file.close()
          self.autoencoder = model_from_json(autoencoder_json)
        else: 
          input_img = Input(shape=(80, 160, 3))

          x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name="conv_1")(input_img)
          x = MaxPooling2D((2, 2), border_mode='same', name="max_pool_1")(x)
          x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name="conv_2")(x)
          x = MaxPooling2D((2, 2), border_mode='same', name="max_pool_2")(x)
          x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name="conv_3")(x)
          x = MaxPooling2D((2, 2), border_mode='same', name="max_pool_3")(x)
          x = Convolution2D(4, 3, 3, activation='relu', border_mode='same', name="conv_4")(x)
          encoded = MaxPooling2D((2, 2), border_mode='same', name="max_pool_4")(x)
        
          x = Convolution2D(4, 3, 3, activation='relu', border_mode='same', name="deconv_1")(encoded)
          x = UpSampling2D((2, 2), name="up_samp_1")(x)
          x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name="deconv_2")(x)
          x = UpSampling2D((2, 2), name="up_samp_2")(x)
          x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name="deconv_3")(x)
          x = UpSampling2D((2, 2), name="up_samp_3")(x)
          x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name="deconv_4")(x)
          x = UpSampling2D((2, 2), name="up_samp_4")(x)
          decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', name="deconv_5")(x)
        
          # this model maps an input to its reconstruction
          self.autoencoder = Model(input_img, decoded)
        
        if os.path.isfile("autoencoder.h5"):
            self.autoencoder.load_weights("autoencoder.h5", by_name=True)
            print("Loaded model from disk")
        
        layer_name = "max_pool_4"
        self.encoder = Model(input=self.autoencoder.input,
                             output=self.autoencoder.get_layer(layer_name).output)
        
        #self.encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
                        
        return self.encoder
