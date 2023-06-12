import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K

"""
Class of ArchModels
----------------------------------------------------
[int int], int, int -> object
----------------------------------------------------
Inputs:
input_dim: 2-array with the dimensions of the image.
latent_dim: int wit the dimension of the latent 
    space.
arch: the architecture of ArchModel to be used.
    The default is the arch=1.
in_channels: input channels to be used. That is, if
    you are using a RGB image as input, it has 3 
    input channels. Also, if you use a ndarray 
    with differents variables (wind, temperature, 
    pressure, etc.), you could use each variable as
    a channel.
out_channels: output channels to be computed. In the
    example of a RBG image as input, you can 
    generate a RGB image (3 output channels) or a
    gray-scale image (1 output channel).
----------------------------------------------------
Output:
After initialization, it generates an object with
the ArchModel architecture and methods
"""
class ArchModels():
    def __init__(self, input_dim, arch=1, in_channels=1, out_channels=2, filter1=112, filter2=32, kernel1=2, kernel2=4, activation1="selu", activation2="selu"):
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = None
        if arch==1:
            # logistic regression
            self.without_intermediate_layers()
        elif arch==2:
            # intermediate layers
            self.with_intermediate_layers()
        elif arch==3:
            # simple convolution
            self.simple_conv2d(filter1, filter2, kernel1, kernel2, activation1, activation2)
        elif arch==4:
            # dense architecture
            self.dense_build()
        elif arch==5:
            # default adapted problem 
            self.advanced_conv2d()
        

    # Logistic regression architecture
    def without_intermediate_layers(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], self.in_channels))
        x = layers.Flatten()(input_img)

        # Output
        out = layers.Dense(self.out_channels, activation='softmax')(x)

        # Model
        self.model = keras.Model(input_img, out)
        print(self.model.summary())

    # Architecture with intermediate layers
    def with_intermediate_layers(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], self.in_channels))

        # Intermediate layers
        x = layers.Flatten()(input_img)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)

        # Output
        out = layers.Dense(self.out_channels, activation='softmax')(x)
        
        # Model
        self.model = keras.Model(input_img, out)
        print(self.model.summary())

    # Dense
    def dense_build(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1]))

        #  Intermediate layers
        x = layers.Flatten()(input_img)
        x = layers.Dense(1024, activation='selu')(x)
        x = layers.Dense(512, activation='selu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='selu')(x)
        x = layers.Dense(32, activation='selu')(x)
        x = layers.Dense(16, activation='selu')(x)
        
        # Output
        out = layers.Dense(self.out_channels, activation='softmax')(x)
        
        # Model
        self.model = keras.Model(input_img, out)
        print(self.model.summary())

    # Simple Convolution
    def simple_conv2d(self, filter1, filter2, kernel1, kernel2, activation1, activation2):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], 1))

        # Conv
        x = layers.Conv2D(filters=filter1, kernel_size=kernel1, activation=activation1, padding='same', strides=1)(input_img)
        x = layers.Conv2D(filters=filter2, kernel_size=kernel2, activation=activation2, padding='same', strides=1)(x)
        x = layers.Flatten()(x)

        # Output
        out = layers.Dense(self.out_channels, activation='softmax')(x)
        
        # Model
        self.model = keras.Model(input_img, out)
        print(self.model.summary())

    # Advanced Convolution
    def advanced_conv2d(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], 1))

        # Conv
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(input_img)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)

        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)

        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(16, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)

        x = layers.Flatten()(x)

        # Output
        out = layers.Dense(self.out_channels, activation='softmax')(x)
        
        # Model
        self.model = keras.Model(input_img, out)
        print(self.model.summary())


    """
    Compilation of the ArchModel.
    Here we specify the method to optimize, the loss function and,
    if we want, some metrics to show together with the loss.
    --------------------------------------------------------------
    string, string, string|array of strings -> 
    --------------------------------------------------------------
    Inputs:
    optimizer: the method to use for optimization. It sould be one
        of tensorflow.keras.optimizers.
    loss: loss function to be optmized. It sould be one of 
        tensorflow.keras.losses.
    metrics: list of metrics to be evaluated by the model during 
        training and testing. It sould be one of
        tensorflow.keras.losses.
    --------------------------------------------------------------
    Outputs:
    No output, only compile the model.
    """
    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['mse', 'mae', 'mape']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    """
    Training of the ArchModel.
    Here we specify the parameters for our training.
    --------------------------------------------------------------
    ndarray, ndarray, int, int, bool, float, int, 
        float, int, bool -> History
    --------------------------------------------------------------
    Inputs:
    x: input data.
    y: output data (data to be compared and trained).
    batch_size: int or None. Number of samples per epoch (gradient
        update).
    shuffle: boolean whether to shuddle the training data before
        each epoch.
    validation_split: float between 0 and 1. Fraction of the 
        training data to be used as validation data.
    verbose: 'auto', 0, 1 or 2. Differents modes of verbosity when
        the model is trained.
    min_delta: minimum change in the monitored quantity to qualify
        as an improvement.
    patience: number of epochs with no improvement adter wich
        training will be stoped.
    respore_best_weigths: boolean whether to restore model weights
        from the epoch with the best value of the monitored
        quantity
    ---------------------------------------------------------------
    Outputs:
    A History object. See History.history.
    """
    def fit(self, x, y, epochs=100, batch_size=64, shuffle=True, validation_split=0.15, verbose=1, min_delta=1e-7, patience=10, restore_best_weights=True):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience,
                                    restore_best_weights=restore_best_weights)
        history = self.model.fit(x, y,
                            epochs=epochs,
                            shuffle=shuffle,
                            validation_split=validation_split,
                            batch_size=batch_size,
                            callbacks=callback,
                            verbose=verbose
                            )
        return history
