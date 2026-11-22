import matplotlib.pyplot as plt

from image_io import Image
from display import Display
from pyramid import GaussianPyramid, LaplacianPyramid
from gui import GUI

def get_greyscale(source=True):
    image = Image(source)
    filename = image.choose_file()
    img_arr = image.to_array(filename)
    grey_img_arr = image.greyscale(img_arr)
    return grey_img_arr

def main():
    """
    The main control flow of OpenGauss
    """

    grey_source_arr = get_greyscale(source=True)
    grey_target_arr = get_greyscale(source=False)

    gui = GUI(grey_source_arr, grey_target_arr)
    gui.build_layout()
    gui.setup_widgets()
    gui.run()

if __name__ == "__main__":
    main()