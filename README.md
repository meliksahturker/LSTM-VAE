# LSTM-VAE
Deep LSTM Variational AutoEncoder

Implemented in Keras.

Most of the implementations on the internet are either wrong, or they do not work with batch size greater than 1.

This one has right implementation and cost function for batch training.

Contains 2 Dense layers at the end to be able to produce high absolute values, since LSTM activations are tanh.
