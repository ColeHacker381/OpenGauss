from transform import Transform
from mask import Mask
from pyramid import GaussianPyramid, LaplacianPyramid

class Blend:

    def __init__(self, levels=5):
        self.gaussian = GaussianPyramid(levels)
        self.laplacian = LaplacianPyramid(levels)
        self.transform = Transform()
        self.mask_generator	= Mask()
        self.levels	= levels

    def blend(self, source, target, source_center, target_center, target_bounds, scale, mask_type):
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

        if mask_type == "ellipse":
            selection_mask = self.mask_generator.create_ellipse(h, w, target_center, (bound_width / 2, bound_height / 2))
        else:
            selection_mask = self.mask_generator.create_rectangle(h, w, target_bounds)

        mask = selection_mask * validity_mask

        aligned_source_gaus_pyramid = self.gaussian.build(aligned_source)
        target_gaus_pyramid = self.gaussian.build(target)
        mask_pyramid = self.gaussian.build(mask)

        aligned_source_lap_pyramid = self.laplacian.build(aligned_source_gaus_pyramid)
        target_lap_pyramid = self.laplacian.build(target_gaus_pyramid)

        blended_pyramid = self.blend_pyramids(aligned_source_lap_pyramid, target_lap_pyramid, mask_pyramid)

        reconstructed = self.laplacian.reconstruct(blended_pyramid)

        final_image = reconstructed[0]
        return final_image

    
    def blend_pyramids(self, source_lap, target_lap, mask_gaus):
        blended = {}

        for level in range(self.levels):
            source = source_lap[level]
            target = target_lap[level]
            mask = mask_gaus[level]

            blended[level] = (
                mask * source + (1.0 - mask) * target
            )

        return blended




