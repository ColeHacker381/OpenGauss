import matplotlib.pyplot as plt
import math

class Display:
    """
    Display grayscale images and intermediate processing results.
    
    Visualization can be enabled or disabled independently
    to assist with debugging the image processing pipeline.
    """

    def __init__(self):
        self.show_greyscale = False
        self.show_blur = False
        self.show_pyramid = True

    def blur(self, source, target):
        """
        Display two images for comparison after filtering.

        Args:
            source: First image to display.
            target: Second image to display.
        """

        if not self.show_blur:
            return
        
        fig, (axis1,axis2) = plt.subplots(1, 2)
        axis1.imshow(source, cmap ="gray", vmin=0, vmax=1)
        axis2.imshow(target, cmap ="gray", vmin=0, vmax=1)
        plt.show()

    def greyscale(self, source, target):
        """
        Display the source and target grayscale images side by side.

        Parameters:
            source: Normalized grayscale source image.
            target: Normalized grayscale target image.
        """

        if not self.show_greyscale:
            return
        
        fig, (axis1,axis2) = plt.subplots(1, 2)
        axis1.imshow(source, cmap ="gray", vmin=0, vmax=1)
        axis2.imshow(target, cmap ="gray", vmin=0, vmax=1)
        plt.show()

    def pyramid(self, source_images, target_images):
        """
        Display two image pyramids for visual comparison.
        Each row represents one image, and each column represents
        a pyramid level from the original to lower resolutions.

        Parameters:
            source_images: Pyramid dictionary of source image.
            target_images: Pyramid dictionary of target image.
        """

        if not self.show_pyramid:
            return

        n_images = len(source_images)
        fig, axes = plt.subplots(2, n_images, figsize=(15, 5))

        #Plot images from source in the first row
        for ax, (key, img) in zip(axes[0], source_images.items()):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f'Source Image: {key}')
            ax.axis('off')

        #Plot images from target in the second row
        for ax, (key, img) in zip(axes[1], target_images.items()):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f'Target Image: {key}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
