
import numpy as np

class Mask:

    def __init__(self):
        pass

    def create_rectangle(self, height, width, bounds):
        xmin, xmax, ymin, ymax = bounds

        mask = np.zeros((height, width), dtype=int)
        mask[ymin:ymax, xmin:xmax] = 1
        return mask

    def create_ellipse(self, height, width, center_coords, axes):

        cx, cy = center_coords
        rx, ry = axes
        y = np.arange(height)[:, np.newaxis]
        x = np.arange(width)[np.newaxis, :]

        mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        return mask.astype(np.float64)

        
