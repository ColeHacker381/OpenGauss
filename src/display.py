import matplotlib.pyplot as plt

class Display:

    def __init__(self):
        self.show_greyscale = True
        self.show_blur = True

    def greyscale(self, source, target):

        if not self.show_greyscale:
            return
        
        fig, (axis1,axis2) = plt.subplots(1, 2)
        axis1.imshow(source, cmap ="gray")
        axis2.imshow(target, cmap ="gray")
        plt.show()

    def blur(self, source, target):
        
        if not self.show_blur:
            return
        
        fig, (axis1,axis2) = plt.subplots(1, 2)
        axis1.imshow(source, cmap ="gray")
        axis2.imshow(target, cmap ="gray")
        plt.show()