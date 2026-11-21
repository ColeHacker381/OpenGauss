import matplotlib.pyplot as plt
import math

class Display:

    def __init__(self):
        self.show_greyscale = False
        self.show_blur = False
        self.show_pyramid = True

    def greyscale(self, source, target):

        if not self.show_greyscale:
            return
        
        fig, (axis1,axis2) = plt.subplots(1, 2)
        axis1.imshow(source, cmap ="gray", vmin=0, vmax=1)
        axis2.imshow(target, cmap ="gray", vmin=0, vmax=1)
        plt.show()

    def blur(self, source, target):
        
        if not self.show_blur:
            return
        
        fig, (axis1,axis2) = plt.subplots(1, 2)
        axis1.imshow(source, cmap ="gray", vmin=0, vmax=1)
        axis2.imshow(target, cmap ="gray", vmin=0, vmax=1)
        plt.show()

    def pyramid(self, source_images, target_images):
        if not self.show_pyramid:
            return

        n_images = len(source_images)  # Assuming both dicts have the same number of images
        fig, axes = plt.subplots(2, n_images, figsize=(15, 5))

        # Plot images from dict1 in the first row
        for ax, (key, img) in zip(axes[0], source_images.items()):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f'Source Image: {key}')
            ax.axis('off')

        # Plot images from dict2 in the second row
        for ax, (key, img) in zip(axes[1], target_images.items()):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f'Target Image: {key}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
