import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

INPUT_DIM = (64,64,3)

CONV_FILTERS = [32,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32
IM_DIM = 64
BATCH_SIZE = 100
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5
DEPTH = 32
LATENT_DEPTH = 512
K_SIZE = 5
lr=0.0001
#lr=0.0001
E_opt = keras.optimizers.Adam(lr=lr)
G_opt = keras.optimizers.Adam(lr=lr)
D_opt = keras.optimizers.Adam(lr=lr)

inner_loss_coef = 1
normal_coef = 0.1
kl_coef = 0.01



class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon


class VAEModel(Model):
    def __init__(self, encoder, generator, discriminator,r_loss_factor, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator

        #self.decoder = decoder
        self.r_loss_factor = r_loss_factor

    @tf.function
    def train_step(self, data):
        E = self.encoder
        G = self.generator
        D = self.discriminator
        # if isinstance(data, tuple):
        #     data = data[0]
        # with tf.GradientTape() as tape:
        #     z_mean, z_log_var, z = self.encoder(data)
        #     reconstruction = self.decoder(z)
        #     reconstruction_loss = tf.reduce_mean(
        #         tf.square(data - reconstruction), axis = [1,2,3]
        #     )
        #     reconstruction_loss *= self.r_loss_factor
        #     kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        #     kl_loss = tf.reduce_sum(kl_loss, axis = 1)
        #     kl_loss *= -0.5
        #     total_loss = reconstruction_loss + kl_loss
        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # return {
        #     "loss": total_loss,
        #     "reconstruction_loss": reconstruction_loss,
        #     "kl_loss": kl_loss,
        # }

        lattent_r =  tf.random.normal((BATCH_SIZE, LATENT_DEPTH))
        with tf.GradientTape(persistent=True) as tape :
            mean, logsigma, kl_loss, lattent = E(data)
            fake = G(lattent)
            dis_fake,dis_inner_fake = D(fake)
            dis_fake_r,_ = D(G(lattent_r))
            dis_true,dis_inner_true = D(data)

            vae_inner = dis_inner_fake-dis_inner_true
            vae_inner = vae_inner*vae_inner
        
            mean,var = tf.nn.moments(E(data)[0], axes=0)
            var_to_one = var - 1
        
            normal_loss = tf.reduce_mean(mean*mean) + tf.reduce_mean(var_to_one*var_to_one)
        
            kl_loss = tf.reduce_mean(kl_loss)
            vae_diff_loss = tf.reduce_mean(vae_inner)
            f_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake), dis_fake))
            r_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake_r), dis_fake_r))
            t_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(dis_true), dis_true))
            gan_loss = (0.5*t_dis_loss + 0.25*f_dis_loss + 0.25*r_dis_loss)
            vae_loss = tf.reduce_mean(tf.abs(data-fake)) 
            E_loss = vae_diff_loss + kl_coef*kl_loss + normal_coef*normal_loss
            G_loss = inner_loss_coef*vae_diff_loss - gan_loss
            D_loss = gan_loss
    
        E_grad = tape.gradient(E_loss,E.trainable_variables)
        G_grad = tape.gradient(G_loss,G.trainable_variables)
        D_grad = tape.gradient(D_loss,D.trainable_variables)
        del tape
        E_opt.apply_gradients(zip(E_grad, E.trainable_variables))
        G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
        D_opt.apply_gradients(zip(D_grad, D.trainable_variables))
        return [gan_loss, vae_loss, f_dis_loss, r_dis_loss, t_dis_loss, vae_diff_loss, E_loss, D_loss, kl_loss, normal_loss]

    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.generator(latent)



class VAEGAN():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.generator = self.models[2]
        self.discriminator = self.models[3]
        #self. = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

    def _build(self):
        # vae_x = Input(shape=INPUT_DIM, name='observation_input')
        # vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='conv_layer_1')(vae_x)
        # vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='conv_layer_2')(vae_c1)
        # vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='conv_layer_3')(vae_c2)
        # vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='conv_layer_4')(vae_c3)

        # vae_z_in = Flatten()(vae_c4)

        # vae_z_mean = Dense(Z_DIM, name='mu')(vae_z_in)
        # vae_z_log_var = Dense(Z_DIM, name='log_var')(vae_z_in)

        # vae_z = Sampling(name='z')([vae_z_mean, vae_z_log_var])
        

        # #### DECODER: 
        # vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        # vae_dense = Dense(1024, name='dense_layer')(vae_z_input)
        # vae_unflatten = Reshape((1,1,DENSE_SIZE), name='unflatten')(vae_dense)
        # vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0], name='deconv_layer_1')(vae_unflatten)
        # vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1], name='deconv_layer_2')(vae_d1)
        # vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2], name='deconv_layer_3')(vae_d2)
        # vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3], name='deconv_layer_4')(vae_d3)
        

        #### MODELS

        vae_encoder = self.encoder()
        vae_generator = self.generator()
        vae_discriminator = self.discriminator()
        #vae_encoder = Model(vae_x, [vae_z_mean, vae_z_log_var, vae_z], name = 'encoder')
        #vae_decoder = Model(vae_z_input, vae_d4, name = 'decoder')

        #vae_full = VAEModel(vae_encoder, vae_decoder, 10000)
        vae_full = VAEModel(vae_encoder, vae_generator, vae_discriminator,10000)

        #opti = Adam(lr=LEARNING_RATE)
        # vae_encoder.compile(optimizer=E_opt)
        # vae_generator.compile(optimizer=G_opt)
        # vae_discriminator.compile(optimizer=D_opt)
        #vae_full.compile(optimizer=opti)
        
        return (vae_full,vae_encoder, vae_generator, vae_discriminator)
    def sampling(self,args):
        mean, logsigma = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
        return mean + tf.exp(logsigma / 2) * epsilon

    def encoder(self):
        input_E = keras.layers.Input(shape=(IM_DIM, IM_DIM, 3))
    
        X = keras.layers.Conv2D(filters=DEPTH*2, kernel_size=K_SIZE, strides=2, padding='same')(input_E)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2D(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(LATENT_DEPTH)(X)    
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
        mean = keras.layers.Dense(LATENT_DEPTH,activation="tanh")(X)
        logsigma = keras.layers.Dense(LATENT_DEPTH,activation="tanh")(X)
        latent = keras.layers.Lambda(self.sampling, output_shape=(LATENT_DEPTH,))([mean, logsigma])
    
        kl_loss = 1 + logsigma - keras.backend.square(mean) - keras.backend.exp(logsigma)
        kl_loss = keras.backend.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
    
        return keras.models.Model(input_E, [mean, logsigma, kl_loss, latent,])
    def generator(self):
        input_G = keras.layers.Input(shape=(LATENT_DEPTH,))

        X = keras.layers.Dense(8*8*DEPTH*8)(input_G)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        X = keras.layers.Reshape((8, 8, DEPTH * 8))(X)
    
        X = keras.layers.Conv2DTranspose(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2DTranspose(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
        X = keras.layers.Conv2DTranspose(filters=DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
        X = keras.layers.Conv2D(filters=3, kernel_size=K_SIZE, padding='same')(X)
        X = keras.layers.Activation('sigmoid')(X)

        return keras.models.Model(input_G, X)

    def discriminator(self):
        input_D = keras.layers.Input(shape=(IM_DIM, IM_DIM, 3))
    
        X = keras.layers.Conv2D(filters=DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
        X = keras.layers.Conv2D(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        X = keras.layers.BatchNormalization()(X)

        X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, padding='same')(X)
        inner_output = keras.layers.Flatten()(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(DEPTH*8)(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
        output = keras.layers.Dense(1)(X)    
    
        return keras.models.Model(input_D, [output, inner_output])
    def set_weights(self, filepath):
        #self.full_model.load_weights(filepath)
        self.encoder.load_weights(filepath+"_encoder")
        self.generator.load_weights(filepath+"_generator")
        self.discriminator.load_weights(filepath+"_discriminator")

    def train(self, data):
        metrics = []
        result = self.full_model.train_step(data)
        print(result)
        # for metric,result in zip(metrics, results) :
        #     metric(result)
        # self.full_model.fit(data, data,
        #         shuffle=True,
        #         epochs=1,
        #         batch_size=BATCH_SIZE)
    
    def save_weights(self, filepath):
        self.encoder.save_weights(filepath+"_encoder")
        self.generator.save_weights(filepath+"_generator")
        self.discriminator.save_weights(filepath+"_discriminator")

        #self.full_model.save_weights(filepath)
