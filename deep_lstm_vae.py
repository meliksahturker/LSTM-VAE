from tensorflow import keras, shape
import tensorflow.keras.backend as K
from tensorflow.keras.layers import(LSTM, Dense, Input, Lambda, RepeatVector, TimeDistributed)
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model

def create_lstm_vae_model(time_steps, number_of_features, int_dim, latent_dim):
    def vae_sampling(args):
        z_mean, z_log_sigma = args
        batch_size = shape(z_mean)[0]
        latent_dim = shape(z_mean)[1]
        epsilon = K.random_normal(shape = (batch_size, latent_dim), mean = 0, stddev = 1)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon
    
    # Encoder
    input_x = Input(shape = (time_steps, number_of_features))
    encoder_LSTM_int = LSTM(int_dim, return_sequences = True)(input_x)
    encoder_LSTM_latent = LSTM(latent_dim, return_sequences = False)(encoder_LSTM_int)

    z_mean = Dense(latent_dim)(encoder_LSTM_latent)
    z_log_sigma = Dense(latent_dim)(encoder_LSTM_latent)
    z_encoder_output = Lambda(vae_sampling, output_shape = (latent_dim,))([z_mean, z_log_sigma])

    encoder = Model(input_x, [z_mean, z_log_sigma, z_encoder_output])

    # Decoder
    decoder_input = Input(shape = (latent_dim))
    decoder_repeated = RepeatVector(time_steps)(decoder_input)
    decoder_LSTM_int = LSTM(int_dim, return_sequences = True)(decoder_repeated)
    decoder_LSTM = LSTM(number_of_features, return_sequences = True)(decoder_LSTM_int)
    decoder_output = TimeDistributed(Dense(number_of_features))(decoder_LSTM)
    decoder = Model(decoder_input, decoder_output)

    # VAE
    output = decoder(encoder(input_x)[2])
    lstm_vae = keras.Model(input_x, output, name = 'lstm_vae')

    # Loss
    rec_loss = K.mean(mse(input_x, output)) * number_of_features
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    vae_loss =  rec_loss + kl_loss

    lstm_vae.add_loss(vae_loss)
    
    return encoder, decoder, lstm_vae
