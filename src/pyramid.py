from filter import Filter

class GaussianPyramid:
    """
    Construct a multiresolution Gaussian image pyramid.
    """

    def __init__(self, levels):
        self.levels = levels
        self.filter = Filter(k_len=5)

    def build(self, image):
        """
        Create a dictionary of images that are blurred and downsampled for each level.

        Parameters:
            image: The image used to create the pyramid.

        Returns:
            images: A dictionary of images that are blurred and downsampled with each iteration.
        """

        images = {}
        for level in range(self.levels):
            images[level] = image
            blur_image = self.filter.gaus_blur_fast(image)
            downsample = self.filter.downsample(blur_image)
            image = downsample

        return images

class LaplacianPyramid:
    """
    Construct a multiresolution Laplacian image pyramid.
    """

    def __init__(self, levels):
        self.levels = levels
        self.filter = Filter(k_len=5)

    def build(self, g_images):
        """
        Apply n level gaussian image to expanded n + 1 gaussian image and save in a dictionary.

        Parameters:
            image: The gaussian image dictionary used to create the pyramid.

        Returns:
            images: A dictionary of images that have been applied and expanded.
        """

        images = {}
        for level in range(self.levels):
            if level == self.levels - 1:
                images[level] = g_images[level]
                continue

            h, w = g_images[level].shape
            images[level] = g_images[level] - self.filter.expand(g_images[level + 1], w, h)

        return images

    def reconstruct(self, l_images):
        """
        Apply n - 1 level laplacian image to expanded n laplacian image and save in a dictionary.

        Parameters:
            image: The laplacian image dictionary used to create the pyramid.

        Returns:
            images: A dictionary of images that have been applied and expanded.
        """ 

        images = {}
        for level in range(self.levels - 1, 0, -1):
            if level == self.levels - 1:
                images[level] = l_images[level]

            h, w = l_images[level - 1].shape
            images[level - 1] = l_images[level - 1] + self.filter.expand(images[level], w, h)

        return images
            
