from transform import Transform
from mask import Mask
from pyramid import GaussianPyramid, LaplacianPyramid

class Blend:
    """
    Perform multiresolution image blending using Gaussian and
    Laplacian pyramids.

    The source image is first aligned with the target image.
    A selection mask determines which regions of each image
    contribute to the final composite.

    Image details are blended at multiple resolutions before
    reconstructing the final image.
    """

    def __init__(self, levels=5):
        self.gaussian = GaussianPyramid(levels)
        self.laplacian = LaplacianPyramid(levels)
        self.transform = Transform()
        self.mask_generator	= Mask()
        self.levels	= levels

    def blend(self, source, target, source_center, target_center, target_bounds, scale, mask_type):
        """
        Blend a selected source region into the target image.

        Paramters:
            source: Normalized grayscale source image.
            target: Normalized grayscale target image.
            source_center: Center of the selected source region (x, y).
            target_center: Desired center in the target image (x, y).
            target_bounds: Bounding rectangle of the destination region.
            scale: Uniform scaling factor applied to the source.
            mask_type: Selection shape ("rectangle" or "ellipse").

        Returns:
            A reconstructed grayscale image matching the target size.
        """

        #Align the source image with the target coordinates
        #The validity mask identifies pixels with valid source data
        aligned_source, validity_mask = self.transform.align(
            source,
            target,
            source_center,
            target_center,
            scale
        )
        h, w = target.shape
        xmin, xmax, ymin, ymax = target_bounds
        bound_width = xmax - xmin
        bound_height = ymax - ymin

        #Create a binary selection mask in the given target coordinates
        if mask_type == "ellipse":
            selection_mask = self.mask_generator.create_ellipse(h, w, target_center, (bound_width / 2, bound_height / 2))
        else:
            selection_mask = self.mask_generator.create_rectangle(h, w, target_bounds)
        
        #Get rid of regions where the transformed source has no pixels
        mask = selection_mask * validity_mask

        #Build Gaussian pyramids for both images and the mask
        #The mask is progressively smoothed at coarser resolutions
        aligned_source_gaus_pyramid = self.gaussian.build(aligned_source)
        target_gaus_pyramid = self.gaussian.build(target)
        mask_pyramid = self.gaussian.build(mask)

        #Convert the Gaussian image pyramids into Laplacian pyramids
        #Each level stores image details missing from the next
        #lower-resolution Gaussian level
        aligned_source_lap_pyramid = self.laplacian.build(aligned_source_gaus_pyramid)
        target_lap_pyramid = self.laplacian.build(target_gaus_pyramid)

        #Combine corresponding Laplacian levels using the
        #Gaussian mask as a varying blender
        blended_pyramid = self.blend_pyramids(aligned_source_lap_pyramid, target_lap_pyramid, mask_pyramid)
        
        #Reconstruct the full resolution image by expanding and
        #adding the blended Laplacian levels
        reconstructed = self.laplacian.reconstruct(blended_pyramid)
        final_image = reconstructed[0]

        return final_image

    def blend_pyramids(self, source_lap, target_lap, mask_gaus):
        """
        Use the Laplacian pyramid images to blend the source and target using the gaussian mask

        Parameters:
            source_lap: The source dictionary of laplacian images
            target_lap: The target dictionary of laplacian images
            mask_gaus: The mask dictionary of gaussian images
        Returns:
            blended: A dictionary of the blended images
        """

        blended = {}
        for level in range(self.levels):
            source = source_lap[level]
            target = target_lap[level]
            mask = mask_gaus[level]

            blended[level] = (
                mask * source + (1.0 - mask) * target
            )

        return blended




