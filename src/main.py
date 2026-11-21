import matplotlib.pyplot as plt

from image_io import Image
from display import Display
from filter import Filter



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
    filter = Filter()

    grey_source_arr = get_greyscale(source=True)
    grey_target_arr = get_greyscale(source=False)
    display.greyscale(grey_source_arr, grey_target_arr)

    blur_source = filter.gaus_blur(grey_source_arr)
    blur_target = filter.gaus_blur(grey_target_arr)
    display.blur(blur_source, blur_target)


    

    





if __name__ == "__main__":
    main()