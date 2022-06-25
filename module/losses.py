from keras.losses import MeanSquaredError
from keras.losses import BinaryCrossEntropy
from keras.losses import Reduction
from tensorflow import reduce_mean


class Losses:
    def __init__(self, numReplicas):
        self.numReplicas = numReplicas

    def bce_loss(self, real, pred):
        # Compute binary cross entropy loss without reduction
        BCE = BinaryCrossEntropy(reduction=Reduction.NONE)
        loss = BCE(real, pred)

        # Compute reduced mean over the entire batch
        loss = reduce_mean(loss)*(1.0/self.numReplicas)

        # Return reduced bce loss
        return loss

    def mse_loss(self, real, pred):
        # Compute mean squared error loss without reduction
        MSE = MeanSquaredError(reduction=Reduction.NONE)

        # Comptue reduced mean over the entire batch
        loss = reduce_mean(MSE(real, pred))*(1.0/self.numReplicas)

        # Return reduced mse loss
        return loss
