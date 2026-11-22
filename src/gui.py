import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches

class GUI:

    def __init__(self, source_image, target_image, levels=5, scale=1.0, mask_type="rectangle"):
        #Image Data
        self.source_image = source_image
        self.target_image = target_image
        self.result_image = None
        self.aligned_source = None
        self.mask = None

        #Selection states
        self.source_bounds = None
        self.source_center = None
        self.target_bounds = None
        self.target_center = None
        self.selection_width = None
        self.selection_height = None
        self.mask_type = mask_type
        self.scale = scale
        self.levels = levels

        #matplotlib objects
        self.fig = None
        self.source_ax = None
        self.rect = None

    def build_layout(self):
        
        self.fig, (self.source_ax,self.target_ax) = plt.subplots(1, 2)
        self.source_artist = self.source_ax.imshow(self.source_image, cmap ="gray", vmin=0, vmax=1)
        self.target_artist = self.target_ax.imshow(self.target_image, cmap ="gray", vmin=0, vmax=1)

        self.fig.subplots_adjust(bottom=0.2)
        self.fig.suptitle("OpenGauss")
        self.source_ax.set_title('Grey Source')
        self.target_ax.set_title('Grey Target')

        self.source_ax.axis('off')
        self.target_ax.axis('off')
        plt.show()

    def on_source_select(self, eclick, erelease):
        xmin, xmax, ymin, ymax = self.selector.extents   
        self.source_bounds = (xmin, xmax, ymin, ymax)
        self.selection_width = xmax - xmin
        self.selection_height = ymax - ymin
        self.source_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        plt.draw()

    def on_target_click(self, event):
        if event.inaxes != self.target_ax or self.source_bounds is None:
            return
        
        self.target_center = (event.xdata, event.ydata)

        if self.rect is not None:
            self.rect.remove()

        self.scale_h = self.selection_height * self.scale
        self.scale_w = self.selection_width * self.scale

        
        xmin = event.xdata - self.scale_w/2
        ymin = event.ydata - self.scale_h/2
        xmax = event.xdata + self.scale_w/2
        ymax = event.ydata + self.scale_h/2
        self.target_bounds = (xmin, xmax, ymin, ymax)

        self.rect = patches.Rectangle(
            (xmin, ymin), 
            self.scale_w, 
            self.scale_h, 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )

        self.target_overlay = self.target_ax.add_patch(self.rect)
        plt.draw()


    def setup_widgets(self):
        self.selector = RectangleSelector(
            self.source_ax, 
            self.on_source_select, 
            useblit=True,        # Improves performance
            button=[1],          # Left mouse button only
            minspanx=5,          # Minimum width in pixels
            minspany=5,          # Minimum height in pixels
            spancoords='data', # Interpret minspan in pixels
            interactive=True     # Allows moving/reshaping after creation
        )

        self.rect = patches.Rectangle(
            (1, 2), 
            width=3, 
            height=1.5, 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )        

        self.target_click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_target_click)


        

