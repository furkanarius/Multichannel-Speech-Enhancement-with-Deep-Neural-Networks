# Multichannel Speech Enhancement with Deep Neural Networks - Beamforming with Autoencoders
This project applies an autoencoder deep neural network to the multichannel speech enhancement problem. It takes the problem from dataset generation to the model training.

### Single Channel and Multichannel Dataset Generation
In order to train the model, you need to create a dataset containing the mixture signals and the clean target signals. The dataset is then converted to the magnitude spectrum. You can find use the code snippets in Dataset Generation folder to create your own dataset. Note that you will need to find your own speech dataset and noise dataset. This set ensures the mixture generation and STFT conversion into a structured form.
