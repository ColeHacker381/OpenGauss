import numpy as np
import matplotlib.image as mpimg
import os, sys
from simple_term_menu import TerminalMenu
from pathlib import Path

class Image:

    def __init__(self, source=True):
        self.cwd = Path.cwd()
        self.image_dir = os.path.join(self.cwd, "inputImages")
        self.filetype = "source" if source else "target"


    def choose_file(self):
        files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        if not files:
            print(f"No files found. Please add '.jpeg', '.jpg' or '.png' images to {self.image_dir} to be used.")
            return None
        
        terminal_menu = TerminalMenu(files, title=f"Select a {self.filetype} file:")
        menu_choice_index = terminal_menu.show()

        if menu_choice_index is not None:
            selected_file = files[menu_choice_index]
            extension = Path(selected_file).suffix
            assert extension in [".jpeg", ".jpg", ".png"], "ERROR: Selected file must be a '.jpeg', '.jpg' or '.png' "
            return os.path.join(self.image_dir, selected_file)

        return None

    def to_array(self, filename):
        try:
            img_arr = mpimg.imread(filename)
        except:
            print("\nImage cannot be read. It could be corrupted?")
            sys.exit()

        return img_arr

    def greyscale(self, img_arr):
        r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
        gray= 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(np.uint8)
