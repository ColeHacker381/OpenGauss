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

    def blend(self, gui: GUI):
        aligned_source, validity_mask = Transform.align(
            gui.source_image,
            gui.target_image,
            gui.source_center,
            gui.target_center,
            gui.scale
        )
        h, w = gui.target_image.shape
        selection_mask = Mask.create_rectangle(h, w, gui.target_bounds)
        mask = selection_mask * validity_mask

        aligned_source_gaus_pyramid = GaussianPyramid.build(aligned_source)
        target_gaus_pyramid = GaussianPyramid.build(gui.target_image)
        mask_pyramid = GaussianPyramid.build(mask)

        aligned_source_lap_pyramid = LaplacianPyramid.build(aligned_source_gaus_pyramid)
        target_lap_pyramid = LaplacianPyramid.build(target_gaus_pyramid)

        blended_pyramid = self.blend_pyramids(aligned_source_lap_pyramid, target_lap_pyramid, mask_pyramid)

        reconstructed = LaplacianPyramid.reconstruct(blended_pyramid)

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




