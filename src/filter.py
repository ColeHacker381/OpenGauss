import numpy as np

class Filter:

    def __init__(self, k_len):
        self.kernel = self.gaus_filter_2d(k_len)
        self.kernel_1d = self.gaus_filter_1d(k_len)

    def gaus_filter_1d(self, n):

        #error checking for positive and odd n
        if n < 1 or n % 2 == 0:
            raise ValueError("Kernel size must be positive and odd.")

        half = n // 2
        sigma = n / 6

        coords = np.arange(-half, half + 1, dtype=np.float64)
        kernel = np.exp(-(coords ** 2) / (2 * sigma ** 2))

        return kernel / kernel.sum()

    def gaus_filter_2d(self, n):
        #error checking for positive and odd n
        if n < 1 or n % 2 == 0:
            raise ValueError("Kernel size must be positive and odd.")

        #works at any odd kernel
        half = n // 2
        sigma = n / 6

        #stolen from wikipedia
        coords = np.arange(-half, half + 1, dtype=np.float64)
        x, y = np.meshgrid(coords, coords)
        kernel = np.exp( -(x ** 2 + y ** 2) / (2 * sigma ** 2))

        return kernel / kernel.sum()

    def gaus_blur(self, image):
        image = np.asarray(image, dtype=np.float64)
        half = len(self.kernel) // 2
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

    def gaus_blur_seperable(self, image):

        image = np.asarray(image, dtype=np.float64)
        half = len(self.kernel_1d) // 2
        h, w = image.shape

        temp = np.empty_like(image)
        out = np.empty_like(image)

        for y in range(h):
            for x in range(w):
                total = 0.0

                for dx in range(-half, half + 1):
                    nx = min(max(x + dx, 0), w - 1)
                    total += image[y, nx] * self.kernel_1d[dx + half]

                temp[y, x] = total

        for y in range(h):
            for x in range(w):
                total = 0.0

                for dy in range(-half, half + 1):
                    ny = min(max(y + dy, 0), h - 1)
                    total += temp[ny, x] * self.kernel_1d[dy + half]

                out[y, x] = total

        return out

    def gaus_blur_fast(self, image):
        image = np.asarray(image, dtype=np.float64)

        h, w = image.shape
        half = len(self.kernel_1d) // 2

        #Horizontal vectorized pass
        padded = np.pad(
            image,
            ((0, 0), (half, half)),
            mode = "edge"
        )

        temp = np.zeros_like(image)
        for k, weight in enumerate(self.kernel_1d):
            temp += weight * padded[:, k:k+w]

        #Vertical vectorized pass
        padded = np.pad(
            temp,
            ((half, half), (0, 0)),
            mode="edge"
        )

        out = np.zeros_like(image)
        for k, weight in enumerate(self.kernel_1d):
            out += weight * padded[k:k+h, :]

        return out

    def downsample(self, image):
        image = np.asarray(image, dtype=np.float64)
        downsampled = image[::2, ::2]
        return downsampled

    def expand(self, image, target_w, target_h):

        h, w = image.shape
        image = image.astype(np.float64, copy=False)

        # Coordinates in original image
        ys = np.arange(target_h) / 2.0
        xs = np.arange(target_w) / 2.0

        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)

        x0 = np.floor(xs).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(ys).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]

        top = (
            (1 - wx) * image[y0[:, None], x0[None, :]]
            + wx * image[y0[:, None], x1[None, :]]
        )

        bottom = (
            (1 - wx) * image[y1[:, None], x0[None, :]]
            + wx * image[y1[:, None], x1[None, :]]
        )

        return (1 - wy) * top + wy * bottom