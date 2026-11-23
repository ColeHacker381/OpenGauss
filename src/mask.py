
import numpy as np

class Mask:
    """Construct methods for multiple mask types."""

    def __init__(self):
        pass

    def create_ellipse(self, height, width, center_coords, axes):
        """
        Create an ellipse mask using the target image coords

        Parameters:
            height: The target image's height.
            width: The target image's width.
            center_coords: The user defined click coordinates.
            axes: The horizontal/vertical radii of the target ellipse.

        Returns:
            A zero mask of the target image's shape with an ellipse of 1s at
            the given location.
        """

        #Unpack the variables, and create the x and y matricies
        cx, cy = center_coords
        rx, ry = axes
        y = np.arange(height)[:, np.newaxis]
        x = np.arange(width)[np.newaxis, :]

        #Use the ellipse equation to set all the values in the
        #mask area to True, then it is converted to 1s
        mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        return mask.astype(np.float64)

    def create_rectangle(self, height, width, bounds):
        """
        Create a rectangle mask using the target image coords.

        Parameters:
            height: The target image's height.
            width: The target image's width.
            bounds: The target rectangle's tuple of (xmin, xmax, ymin, ymax).

        Returns:
            A zero mask of the target image's shape with a rectangle of 1s at
            the given location.
        """

        #Unpack the variables and convert to integers,
        #ensuring that the x values will be clipped
        #if they are given a negative value
        xmin, xmax, ymin, ymax = bounds
        xmin, xmax = int(np.floor(xmin)), int(np.ceil(xmax))
        xmin, xmax = max(0, min(width, xmin)), max(0, min(width, xmax))
        ymin, ymax = int(np.floor(ymin)), int(np.ceil(ymax))

        #Create a zero mask, and use vector slicing
        #to set all values in the rectangle coords to 1
        mask = np.zeros((height, width), dtype=float)
        mask[ymin:ymax, xmin:xmax] = 1.0
        return mask
        