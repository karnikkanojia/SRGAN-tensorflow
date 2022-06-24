from tensorflow.io import FixedLenFeature
from tensorflow.io import parse_single_example
from tensorflow.io import parse_tensor
from tensorflow.image import flip_left_right
from tensorlow.image import rot90
import tensorflow as tf

# Define AUTOTUNE object
try:
    AUTO = tf.data.AUTOTUNE
except:
    AUTO = tf.data.experimental.AUTOTUNE


def random_crop(lrImage, hrImage, hrCropSize=96, scale=4):
    # Calculate the low resolution image crop size and image shape
    lrCropSize = hrCropSize // scale
    lrImageShape = tf.shape(lrImage)[:2]

    # Calculate the low resolution image width and height offset
    lrW = tf.random.uniform(
        shape=(), maxval=lrImageShape[1] - lrCropSize + 1, dtype=tf.int32)
    lrH = tf.random.uniform(
        shape=(), maxval=lrImageShape[0] - lrCropSize + 1, dtype=tf.int32)

    # Calculate the high resolution image width and height
    hrW = lrW * scale
    hrH = lrH * scale

    # Crop the low and high resolution images
    lrImageCropped = tf.slice(lrImage, [lrH, lrW, 0],
                              [(lrCropSize), (lrCropSize), 3])
    hrImageCropped = tf.slice(hrImage, [hrH, hrW, 0],
                              [(hrCropSize), (hrCropSize), 3])

    return (lrImageCropped, hrImageCropped)


def get_center_crop(lrImage, hrImage, hrCropSize=96, scale=4):
    # Calculate the low resolution image crop size and image shape
    lrCropSize = hrCropSize // scale
    lrImageShape = tf.shape(lrImage)[:2]

    # Calculate the low resolution image width and height
    lrW = lrImageShape[1] // 2
    lrH = lrImageShape[0] // 2

    # Calculate the high resolution image width and height
    hrW = lrW * scale
    hrH = lrH * scale

    # Crop the low and high resolution images
    lrImageCropped = tf.slice(lrImage, [lrH - (lrCropSize // 2),
                                        lrW - (lrCropSize // 2), 0],
                              [lrCropSize, lrCropSize, 3])

    hrImageCropped = tf.slice(hrImage, [hrH - (hrCropSize // 2),
                                        hrW - (hrCropSize // 2), 0],
                              [hrCropSize, hrCropSize, 3])

    return (lrImageCropped, hrImageCropped)


def random_flip(lrImage, hrImage):
    # Calculate a random chance for flip
    flipProb = tf.random.uniform(shape=(), maxval=1)
    (lrImage, hrImage) = tf.cond(flipProb < 0.5,
                                 lambda: (lrImage, hrImage),
                                 lambda: (flip_left_right(lrImage), flip_left_right(hrImage)))

    # Return the randomly flipped low and high resolution images
    return (lrImage, hrImage)


def random_rotate(lrImage, hrImage):
    # Randomly generate number of 90 degree rotations
    n = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)

    # Rotate the low and high resolution images
    lrImage = rot90(lrImage, n)
    hrImage = rot90(hrImage, n)

    # Return the randomly rotated images
    return (lrImage, hrImage)


def read_train_example(example):
    # Parse the example into features and labels
    features = {
        'lr': FixedLenFeature([], tf.string),
        'hr': FixedLenFeature([], tf.string),
    }
    example = parse_single_example(example, features)

    # Parse the features into tensors
    lrImage = parse_tensor(example['lr'], out_type=tf.uint8)
    hrImage = parse_tensor(example['hr'], out_type=tf.uint8)

    # Perform data augmentation
    (lrImage, hrImage) = random_crop(lrImage, hrImage)
    (lrImage, hrImage) = random_flip(lrImage, hrImage)
    (lrImage, hrImage) = random_rotate(lrImage, hrImage)

    # Reshape the low and high resolution images
    lrImage = tf.reshape(lrImage, [24, 24, 3])
    hrImage = tf.reshape(lrImage, [96, 96, 3])

    # Return the low and high resolution images
    return (lrImage, hrImage)


def read_test_example(example):
    # Get the feature template and parse a single image according to the feature examples
    feature = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string),
    }
    example = parse_single_example(example, feature)

    # Parse the low and high resolution images
    lrImage = parse_tensor(example['lr'], out_type=tf.uint8)
    hrImage = parse_tensor(example['hr'], out_type=tf.uint8)

    # Center crop both low and high resolution images
    (lrImage, hrImage) = get_center_crop(lrImage, hrImage)

    # Reshape the low and high resolution images
    lrImage = tf.reshape(lrImage, [24, 24, 3])
    hrImage = tf.reshape(hrImage, [96, 96, 3])

    # Return the low and high resolution images
    return (lrImage, hrImage)


def load_dataset(filenames, batchSize, train=False):
    # Get the TFRecords from the filenames
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    # Check if this is training or testing dataset
    if train:
        # Read the training examples
        dataset = dataset.map(read_train_example, num_parallel_calls=AUTO)

    # Otherwise we are working with the testing dataset
    else:
        # Read the test examples
        dataset = dataset.map(read_test_example, num_parallel_calls=AUTO)

    # Batch and prefetch the dataset
    dataset = (dataset.shuffle(batchSize).batch(
        batchSize).repeat().prefetch(AUTO))

    return dataset
