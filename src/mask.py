
import numpy as np

class Mask:

    def __init__(self):
        pass

    def create_rectangle(self, height, width, bounds):
        xmin, xmax, ymin, ymax = bounds
        xmin, xmax = int(np.floor(xmin)), int(np.ceil(xmax))
        ymin, ymax = int(np.floor(ymin)), int(np.ceil(ymax))

        mask = np.zeros((height, width), dtype=float)
        mask[ymin:ymax, xmin:xmax] = 1.0
        return mask

    def create_ellipse(self, height, width, center_coords, axes):

        cx, cy = center_coords
        rx, ry = axes
        y = np.arange(height)[:, np.newaxis]
        x = np.arange(width)[np.newaxis, :]

        mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        return mask.astype(np.float64)

        
