from keras.layers import Rescaling
from tensorflow._api.v2.nn import depth_to_space
from keras.layers import BatchNormalization, GlobalAvgPool2D, LeakyReLU, Conv2D, Dense, PReLU, Add
from keras import Model, Input


class SRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlock):
        inputs = Input((None, None, 3))
        xIn = Rescaling(scale=(1./255.), offset=0.)(inputs)

        # Pass the input through Conv => BatchNorm => PReLU
        x = Conv2D(featureMaps, (3, 3), padding='same')(xIn)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(featureMaps, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        xSkip = Add()[xIn, x]

        # Creating several residual blocks
        for _ in range(residualBlock-1):
            x = Conv2D(featureMaps, (3, 3), padding='same')(xSkip)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(featureMaps, (3, 3), padding='same')(x)
            xSkip = Add()[xSkip, x]

        # Get the last residual block without activation
        x = Conv2D(featureMaps, (3, 3), padding='same')(xSkip)
        x = BatchNormalization()(x)
        x = Add()[xIn, x]

        # Upscale the image with pixel shuffle
        x = Conv2D(featureMaps*(scalingFactor // 2), (3, 3), padding='same')(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)

        # Upscale the image with pixel shuffle
        x = Conv2D(featureMaps*scalingFactor, (3, 3), padding='same')(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)

        # Get the output and scale it from [-1, 1] to [0, 255] range
        x = Conv2D(3, (3, 3), padding='same', activation='tanh')(x)
        x = Rescaling(scale=127.5, offset=127.5)(x)

        generator = Model(inputs, x)

        # Return the generator
        return generator

    @staticmethod
    def discriminator(featureMaps, leakyAlpha, discBlocks):
        # Initialize the input layer and process it with conv kernel
        inputs = Input((None, None, 3))
        x = Rescaling(scale=(1./127.5), offset=-1.0)(inputs)
        x = Conv2D(featureMaps, (3, 3), padding='same')(x)

        # Unlike the generator, we will use leakyReLU
        x = LeakyReLU(leakyAlpha)(x)

        # Pass the output from previous layer through a Conv => BatchNorm => LeakyReLU
        x = Conv2D(featureMaps, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leakyAlpha)(x)

        for i in range(1, discBlocks):
            # Pass the output from previous layer through a Conv => BatchNorm => LeakyReLU
            x = Conv2D(featureMaps*(2**i), (3, 3),
                       padding='same', strides=2)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)

            # Second Conv2D => BatchNorm => LeakyReLU
            x = Conv2D(featureMaps*(2**i), (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)

        # Process the feature maps with GlobalAvgPool2D
        x = GlobalAvgPool2D()(x)
        x = LeakyReLU(leakyAlpha)(x)

        # Final FC layer with sigmoid activation function
        x = Dense(1, activation='sigmoid')(x)

        discriminator = Model(inputs, x)

        # Return the discriminator
        return discriminator
