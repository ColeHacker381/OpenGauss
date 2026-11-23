import numpy as np

class Filter:
    """
    Apply Gaussian filtering and image resampling operations.
    """

    def __init__(self, k_len):
        self.kernel_1d = self.gaus_kernel_1d(k_len)
        self.kernel_2d = self.gaus_kernel_2d(k_len)

    def downsample(self, image):
        """
        Reduce an image by selecting every second pixel.

        Parameters:
            image: 2D grayscale image.

        Returns:
            Image with about half the original dimensions.
        """

        image = np.asarray(image, dtype=np.float64)
        downsampled = image[::2, ::2]
        return downsampled

    def expand(self, image, target_w, target_h):
        """
        Expand an image to a target size using bilinear interpolation.

        The method maps destination coordinates back to the smaller
        source image and interpolates values from four neighbors.

        Parameters:
            image: 2D grayscale image.
            target_w: Desired output width.
            target_h: Desired output height.

        Returns:
            Expanded image with the shape of (target_h, target_w).
        """

        h, w = image.shape
        image = image.astype(np.float64, copy=False)

        #Map output pixels into the original image coordinates
        ys = np.arange(target_h) / 2.0
        xs = np.arange(target_w) / 2.0

        #Clamp sampling coordinates to valid pixel boundaries
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)

        #Get the four neighboring source pixels
        x0 = np.floor(xs).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(ys).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        #Fractional distances allow for the interpolation weights
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]

        #Interpolate along the upper and lower rows
        top = (
            (1 - wx) * image[y0[:, None], x0[None, :]]
            + wx * image[y0[:, None], x1[None, :]]
        )

        bottom = (
            (1 - wx) * image[y1[:, None], x0[None, :]]
            + wx * image[y1[:, None], x1[None, :]]
        )

        #Interpolate vertically to obtain the final pixel values
        return (1 - wy) * top + wy * bottom
    
    def gaus_blur(self, image):
        """
        Apply Gaussian blur using direct 2D convolution.

        Uses four nested loops to evaluate each kernel position
        for every image pixel.

        Parameters:
            image: 2D grayscale image.

        Returns:
            Gaussian blurred image with the original dimensions.
        """

        image = np.asarray(image, dtype=np.float64)
        half = len(self.kernel) // 2
        h, w = image.shape

        out = np.empty_like(image, dtype=np.float64)
        for y in range(h):
            for x in range(w):
                total = 0.0

                 #Accumulate the weighted surroundings around a pixel
                for dy in range(-half, half + 1):
                    for dx in range(-half, half + 1):

                        #Recreate edge pixels at image boundaries
                        ny = min(max(y + dy, 0), h - 1)
                        nx = min(max(x + dx, 0), w - 1)
                        total += image[ny, nx] * self.kernel[dy + half, dx + half]

                out[y, x] = total
        return out

    def gaus_blur_fast(self, image):
        """
        Apply vectorized Gaussian filtering.

        Uses NumPy array slicing to convolve entire rows and
        columns at the same time, getting rid of python for loops.

        Parameters:
            image: 2D grayscale image.

        Returns:
            Gaussian blurred image with the original dimensions.
        """

        image = np.asarray(image, dtype=np.float64)
        h, w = image.shape
        half = len(self.kernel_1d) // 2

        #Pad columns to preserve the original image dimensions
        padded = np.pad(
            image,
            ((0, 0), (half, half)),
            mode = "edge"
        )

        #Horizontal convolution using shifted array slices
        temp = np.zeros_like(image)
        for k, weight in enumerate(self.kernel_1d):
            temp += weight * padded[:, k:k+w]

        #Pad rows before the vertical convolution
        padded = np.pad(
            temp,
            ((half, half), (0, 0)),
            mode="edge"
        )

        #Vertical convolution using shifted array slices.
        out = np.zeros_like(image)
        for k, weight in enumerate(self.kernel_1d):
            out += weight * padded[k:k+h, :]

        return out

    def gaus_blur_seperable(self, image):
        """
        Apply Gaussian blur using two 1D passes.

        The image is filtered horizontally and then vertically.
        This reduces the work compared with direct 2D convolution.

        Parameters:
            image: 2D grayscale image.

        Returns:
            Gaussian blurred image with the original dimensions.
        """

        image = np.asarray(image, dtype=np.float64)
        half = len(self.kernel_1d) // 2
        h, w = image.shape

        temp = np.empty_like(image)
        out = np.empty_like(image)

        #Convolve each row horizontally
        for y in range(h):
            for x in range(w):
                total = 0.0

                for dx in range(-half, half + 1):
                    nx = min(max(x + dx, 0), w - 1)
                    total += image[y, nx] * self.kernel_1d[dx + half]

                temp[y, x] = total

        #Convolve the intermediate image vertically
        for y in range(h):
            for x in range(w):
                total = 0.0

                for dy in range(-half, half + 1):
                    ny = min(max(y + dy, 0), h - 1)
                    total += temp[ny, x] * self.kernel_1d[dy + half]

                out[y, x] = total

        return out
    
    def gaus_kernel_1d(self, n):
        """
        Generate a normalized 1D Gaussian kernel.

        The standard deviation is approximated as n / 6,
        placing approximately three standard deviations on
        either side of the kernel center.

        Parameters:
            n: Positive odd kernel size.

        Returns:
            1D Gaussian kernel whose weights sum to 1.
        """

        if n < 1 or n % 2 == 0:
            raise ValueError("Kernel size must be positive and odd.")

        half = n // 2
        sigma = n / 6
        coords = np.arange(-half, half + 1, dtype=np.float64)

        #Compute Gaussian weights and normalize their sum to 1
        kernel = np.exp(-(coords ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def gaus_kernel_2d(self, n):
        """
        Generate a normalized 2D Gaussian kernel.

        Parameters:
            n: Positive odd kernel size.

        Returns:
            2D Gaussian kernel whose weights sum to 1.
        """

        if n < 1 or n % 2 == 0:
            raise ValueError("Kernel size must be positive and odd.")

        half = n // 2
        sigma = n / 6

        #Construct a coordinate grid centered on the kernel
        coords = np.arange(-half, half + 1, dtype=np.float64)
        x, y = np.meshgrid(coords, coords)
        
        #Evaluate the 2D Gaussian function at every grid point
        kernel = np.exp( -(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()
