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

    #to control display options, go to display.py
    #do not comment out display calls
    display = Display()
    gaus_pyramid = GaussianPyramid(levels=5)
    lap_pyramid = LaplacianPyramid(levels=5)

    grey_source_arr = get_greyscale(source=True)
    grey_target_arr = get_greyscale(source=False)
    gui = GUI(grey_source_arr, grey_target_arr)
    gui.build_layout()
    # display.greyscale(grey_source_arr, grey_target_arr)

    # source_gaus_pyramid = gaus_pyramid.build(grey_source_arr)
    # target_gaus_pyramid = gaus_pyramid.build(grey_target_arr)
    # display.pyramid(source_gaus_pyramid, target_gaus_pyramid)

    # source_lap_pyramid = lap_pyramid.build(source_gaus_pyramid)
    # target_lap_pyramid = lap_pyramid.build(target_gaus_pyramid)
    # display.pyramid(source_lap_pyramid, target_lap_pyramid)

    # source_lap_pyramid = lap_pyramid.reconstruct(source_lap_pyramid)
    # target_lap_pyramid = lap_pyramid.reconstruct(target_lap_pyramid)
    # display.pyramid(source_lap_pyramid, target_lap_pyramid)


    

    





if __name__ == "__main__":
    main()