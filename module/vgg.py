from keras.applications import VGG19
from keras import Model


class VGG:
    @staticmethod
    def build():
        # Initialize the pre-trained VGG19 model
        vgg = VGG19(weights="imagenet", include_top=False,
                    input_shape=(None, None, 3))

        # Slcing the model till 20th layer
        model = Model(vgg.input, vgg.layers[20].output)

        # Return the sliced VGG19 model
        return model
