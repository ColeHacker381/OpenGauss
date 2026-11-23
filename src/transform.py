import numpy as np

class Transform:
    """
    Perform geometric transformations on grayscale images.
    """

    def __init__(self):
        pass
    
    def align(self, source, target, source_center, target_center, scale):
        """
        Align a source image with a target image.

        Parameters:
            source: Normalized grayscale source image.
            target: Normalized grayscale target image.
            source_center: Selected source image center.
            target_center: Selected destination target center.
            scale: The positive scaling factor.

        Returns:
            aligned_source: 
                Target-sized transformed source image.
            validity_mask: 
                Boolean array telling where the transformed source has valid pixel data.
        """
        #Map every target coordinate into the source coordinate system
        #Record which mapped coordinates contain valid source data
        #Sample the source image at the inverse mapped coordinates
        xs, ys = self.inverse_map(target.shape, source_center, target_center, scale)
        vmask = self.create_validity_mask(xs, ys, source.shape)
        sample_source = self.bilinear_sample(source, xs, ys)

        #Copy the target image, and replace only pixels 
        #corresponding to valid source samples.
        aligned_source = target.copy()
        aligned_source[vmask] = sample_source[vmask]
        return aligned_source, vmask

    def bilinear_sample(self, image, xs, ys):
        """
        Sample an image at fractional coordinates using bilinear interpolation.
        Each output pixel is calculated from the four nearest source pixels.

        Parameters:
            image: 2D grayscale source image.
            xs: 1D source x sampling coordinates.
            ys: 1D source y sampling coordinates.

        Returns:
            Interpolated image with a shape (len(ys), len(xs)).
        """

        h, w = image.shape

        #Clamp to prevent invalid indexing
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)

        #Locate the four neighboring pixels around each sample
        x0 = np.floor(xs).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(ys).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        #Calculate fractional distances between neighboring pixels
        #and enable NumPy broadcasting
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]

        #Interpolate horizontally along the upper/lower edge.
        top = (
            (1 - wx) * image[y0[:, None], x0[None, :]]
            + wx * image[y0[:, None], x1[None, :]]
        )

        bottom = (
            (1 - wx) * image[y1[:, None], x0[None, :]]
            + wx * image[y1[:, None], x1[None, :]]
        )

        #Interpolate vertically between the two horizontal results.
        return (1 - wy) * top + wy * bottom

    def create_validity_mask(self, xs, ys, source_shape):
        """
        Identify target pixels that map inside the source image.

        Parameters:
            xs: Source x sampling coordinates.
            ys: Source y sampling coordinates.
            source_shape: The source image dimensions.

        Returns:
            vmask: 
                Boolean array matching the target dimensions.
                True indicates that a source pixel is available.
        """

        height, width = source_shape

        #Check each coordinate against the source image boundaries
        vx = (xs >= 0) & (xs <= width - 1)
        vy = (ys >= 0) & (ys <= height - 1)

        #Broadcasting combines the horizontal and vertical mask
        #into a 2D target mask
        vmask = vy[:, None] & vx[None, :]

        return vmask

    def inverse_map(self, target_shape, source_center, target_center, scale):
        """
        Map target image coordinates into source image coordinates.
        Uses inverse translation and uniform scaling so that each
        target pixel can sample the correct source location.

        Parameters:
            target_shape: The target image dimensions.
            source_center: The source selection center.
            target_center: The target destination center.
            scale: The positive scaling factor.

        Returns:
            xs: 1D array of source x coordinates.
            ys: 1D array of source y coordinates.
        """

        height, width = target_shape
        csx, csy = source_center
        ctx, cty = target_center

        xt = np.arange(width)
        yt = np.arange(height)

        #Each output coordinate maps back to its source position.
        xs = (xt - ctx) / scale + csx
        ys =  (yt - cty) / scale + csy

        return xs, ys
