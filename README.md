# LSTM-VAE
Deep LSTM Variational AutoEncoder

![Deep LSTM-VAE](https://user-images.githubusercontent.com/67103746/113914558-589eb380-97e6-11eb-881e-e106f83cb66d.png)


Implemented in Keras.

Most of the implementations on the internet are either wrong, or they do not work with batch size greater than 1, because their loss function is wrong.

This one has right implementation and cost function for batch training.

Contains Dense layer at the end to be able to produce high absolute values, since LSTM activations are tanh.
