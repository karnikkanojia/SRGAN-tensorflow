from keras import Model
from tensorflow import GradientTape, concat, zeros, ones
import tensorflow as tf
import keras


class SRGANTraining(Model):
    def __init__(self, generator, discriminator, vgg, batchSize):
        super().__init__()
        # Initialization of generator and discriminator, vggmodel and global batch size
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.batchSize = batchSize

    def compile(self, gOptimizer, dOptimizer, bceLoss, mseLoss):
        super().compile()
        self.gOptimizer = gOptimizer
        self.dOptimizer = dOptimizer

        # Initialize the loss functions
        self.bceLoss = bceLoss
        self.mseLoss = mseLoss

    def train_step(self, images):
        # Grab the low and high resoltuion images
        (lrImages, hrImages) = images
        lrImages = tf.cast(lrImages, tf.float32)
        hrImages = tf.cast(hrImages, tf.float32)

        # Generate the high resolution images
        srImages = self.generator(lrImages)

        # Combine them with real images
        combinedImages = concat([srImages, hrImages], axis=0)

        # Assemble the labels discriminating the real ones from the fake ones label 0 is for predicted images and 1 is for high resol
        labels = concat(
            [zeros((self.batchSize, 1)), ones(self.batchSize, 1)], axis=0)

        # Train the discriminator
        with GradientTape() as tape:
            # Get the discriminator predictions
            predictions = self.discriminator(combinedImages)

            # Compute the loss
            dLoss = self.bceLoss(labels, predictions)

        grads = tape.gradient(dLoss, self.discriminator.trainable_variables)

        # Optimize the discriminator weights according to the gradients computed
        self.dOptimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )

        # Generate misleading labels
        misleadingLabels = ones((self.batchSize, 1))

        # Train the generator

        with GradientTape() as tape:
            # Get fake images from generator
            fakeImages = self.generator(lrImages)

            # Get the predictions from the discriminator
            predictions = self.discriminator(fakeImages)

            # Compute the adversarial loss
            gLoss = 1e-3*self.bceLoss(misleadingLabels, predictions)

            # Compute the normalized vgg outputs
            srVgg = keras.applications.vgg19.preprocess_input(fakeImages)
            srVgg = self.vgg(srVgg)/12.75
            hrVgg = keras.applications.vgg19.preprocess_input(hrImages)

            # Compute the perceptual loss
            percLoss = self.mseLoss(hrVgg, srVgg)

            # Calculate the total generator loss
            gTotalLoss = gLoss + percLoss

        grads = tape.gradient(gTotalLoss, self.generator.trainable_variables)

        # Optimize the generator weights according to the gradients computed
        self.gOptimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {
            "dLoss": dLoss,
            "gLoss": gLoss,
            "percLoss": percLoss,
            "gTotalLoss": gTotalLoss
        }
