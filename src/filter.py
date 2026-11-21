import numpy as np

class Filter:

    def __init__(self):
        self.kernel = self.gaus_filter(5)

    def gaus_filter(self, n):
        #works at any odd kernal
        half = (n - 1) // 2
        sigma = n / 6

        #stolen from wikipedia
        coords = np.arange(-half, half + 1)
        x, y = np.meshgrid(coords, coords)
        k = np.exp( -(x ** 2 + y ** 2) / (2 * sigma ** 2))

        return k / k.sum()

    def gaus_blur(self, image):

        half = self.kernel.shape[0] // 2
        h, w = image.shape
        out = np.empty_like(image, dtype=np.float64)
        for y in range(h):
            for x in range(w):
                total = 0.0
                for dy in range(-half, half + 1):
                    for dx in range(-half, half + 1):
                        ny = min(max(y + dy, 0), h - 1)
                        nx = min(max(x + dx, 0), w - 1)
                        total += image[ny, nx] * self.kernel[dy + half, dx + half]

                out[y, x] = total
        return out
