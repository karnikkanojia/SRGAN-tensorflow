from . import config
from matplotlib.pyplot import subplots
from matplotlib.pyplot import savefig
from matplotlib.pyplot import title
from matplotlib.pyplot import xticks
from matplotlib.pyplot import yticks
from matplotlib.pyplot import show
from keras_preprocessing.image import array_to_img
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os


def zoom_into_images(image, imageTitle):
    # Create a new figure with a default 111 subplots
    (fig, ax) = subplots()
    im = ax.imshow(array_to_img(image[::-1]), origin="lower")

    title(imageTitle)
    # Zoom factor: 2.0, location -> upper-left
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(array_to_img(image[::-1]), origin="lower")

    # Specify the limits
    (x1, x2, y1, y2) = 20, 40, 20, 40
    # Apply the x-limits
    axins.set_xlim(x1, x2)
    # Apply the y-limits
    axins.set_ylim(y1, y2)

    # Remove the xticks and yticks
    xticks(visible=False)
    yticks(visible=False)

    # Make the line
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")

    # Build the image path and save it to disk
    imagePath = os.path.join(config.BASE_IMAGE_PATH, f"{imageTitle}.png")
    savefig(imagePath)

    # Show the image
    show()
