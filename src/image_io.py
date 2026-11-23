import numpy as np
import matplotlib.image as mpimg
import os, sys
from simple_term_menu import TerminalMenu
from pathlib import Path

class Image:
    """
    Constructs methods for selecting, reading, and greyscaling an image.
    """

    def __init__(self, source: bool):
        self.cwd = Path.cwd()
        self.image_dir = os.path.join(self.cwd, "inputImages")
        self.filetype = "source" if source else "target"

    def choose_file(self):
        """
        Creates a small CLI that allows for easy choosing of image to be read.

        Returns:
            filename: The filename of the image to use.
        """

        #Scrapes each image out of the image input directory
        files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        if not files:
            print(f"No files found. Please add '.jpeg', '.jpg' or '.png' images to {self.image_dir} to be used.")
            return None

        #Creates the small CLI of images to choose from
        terminal_menu = TerminalMenu(files, title=f"Select a {self.filetype} file:")
        menu_choice_index = terminal_menu.show()

        #Ensures the selected file is an image, and returns the filepath
        if menu_choice_index is not None:
            selected_file = files[menu_choice_index]
            extension = Path(selected_file).suffix
            assert extension in [".jpeg", ".jpg", ".png"], "ERROR: Selected file must be a '.jpeg', '.jpg' or '.png' "
            return os.path.join(self.image_dir, selected_file)

        return None

    def greyscale(self, img_arr):
        """
        Takes an RGB/RGBA image array and converts it to greyscale.

        Parameters:
            img_arr: The 3D/4D image array to be converted.

        Returns:
            grey_arr: The greyscaled 2D image array.
        """

        #handle greyscale and unsupported images
        if img_arr.ndim == 2:
            return img_arr.astype(np.float32)
        elif img_arr.ndim != 3 or img_arr.shape[2] < 3:
            raise ValueError("Unsupported image shape.")

        #if rgba, need to get rid of alpha channel
        rgb = img_arr[:, :, :3]

        #scale RGB image to greyscale using channel math conversion
        r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
        gray= 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.float32)

    def to_array(self, filename):
        """
        Reads in an image and converts it to 0-1 float.

        Parameters:
            filename: The file to be read in.

        Returns:
            img_arr: The image read in as an array.
        """

        #Try to read in the file
        try:
            img_arr = mpimg.imread(filename)
        except (OSError, ValueError) as e:
            print(f"Image {filename} cannot be read.")
            sys.exit()

        #convert integers to 0-1 normalized floats
        original_dtype = img_arr.dtype
        img_arr = img_arr.astype(np.float32)
        if np.issubdtype(original_dtype, np.integer):
            maximum = np.iinfo(original_dtype).max
            img_arr /= maximum

        return img_arr
