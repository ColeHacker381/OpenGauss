import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, EllipseSelector, Button, Slider, RadioButtons
import matplotlib.patches as patches
from blend import Blend

class GUI:

    def __init__(self, source_image, target_image, levels=5, scale=1.0, mask_type="rectangle"):
        #Image Data
        self.source_image = source_image
        self.target_image = target_image
        self.result_image = target_image
        self.aligned_source = None
        self.mask = None

        #Selection states
        self.source_bounds = None
        self.source_center = None
        self.target_bounds = None
        self.target_center = None
        self.target_overlay = None
        self.selection_width = None
        self.selection_height = None
        self.mask_type = mask_type
        self.scale = scale
        self.scale_slider = None
        self.levels = levels

        #matplotlib objects
        self.fig = None
        self.source_ax = None
        self.rect = None

        #class objects
        self.blender = Blend(self.levels)

    def build_layout(self):
        
        self.fig, (self.source_ax,self.target_ax, self.result_ax) = plt.subplots(1, 3)
        self.source_artist = self.source_ax.imshow(self.source_image, cmap ="gray", vmin=0, vmax=1)
        self.target_artist = self.target_ax.imshow(self.target_image, cmap ="gray", vmin=0, vmax=1)
        self.result_artist = self.result_ax.imshow(self.result_image, cmap ="gray", vmin=0, vmax=1)

        self.fig.subplots_adjust(bottom=0.3)
        self.fig.suptitle("OpenGauss")
        self.source_ax.set_title('Grey Source')
        self.target_ax.set_title('Grey Target')
        self.target_ax.set_title('Grey Result')

        self.source_ax.axis('off')
        self.target_ax.axis('off')
        self.result_ax.axis('off')

    def setup_source_selector(self):
        if self.mask_type == "ellipse":
            self.selector = self.ellipse_selector
        else:
            self.selector = self.rectangle_selector

        return self.selector

    def on_source_select(self, eclick, erelease):
        xmin, xmax, ymin, ymax = self.selector.extents   
        self.source_bounds = (xmin, xmax, ymin, ymax)
        self.selection_width = xmax - xmin
        self.selection_height = ymax - ymin
        self.source_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        self.fig.canvas.draw_idle()

    def on_target_click(self, event):
        if event.inaxes != self.target_ax:
            return

        if self.source_bounds is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        self.target_center = (event.xdata, event.ydata)
        self.update_preview()

    def on_scale_change(self, value):
        self.scale = value
        self.update_preview()

    def on_mask_change(self, label):
        self.mask_type = label.lower()
        self.setup_source_selector()
        self.update_preview()

    def on_scroll(self, event):

        if event.xdata is None or event.ydata is None:
            return

        ax = event.inaxes
        zoom_factor = 1.2

        if event.button == "up":
            scale = 1 / zoom_factor
        elif event.button == "down":
            scale = zoom_factor
        else:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x = event.xdata
        y = event.ydata

        ax.set_xlim(
            x + (xlim[0] - x) * scale,
            x + (xlim[1] - x) * scale
        )
        ax.set_ylim(
            y + (ylim[0] - y) * scale,
            y + (ylim[1] - y) * scale
        )
        self.fig.canvas.draw_idle()


    def update_preview(self):

        if self.target_center is None:
            return

        self.scale_h = self.selection_height * self.scale
        self.scale_w = self.selection_width * self.scale

        ctx, cty = self.target_center
        
        xmin = ctx - self.scale_w/2
        ymin = cty - self.scale_h/2
        xmax = ctx + self.scale_w/2
        ymax = cty + self.scale_h/2
        self.target_bounds = (xmin, xmax, ymin, ymax)

        # Remove previous preview
        if self.target_overlay is not None:
            self.target_overlay.remove()

        if self.mask_type == "ellipse":
            patch = patches.Ellipse(
                (ctx, cty),
                width=self.scale_w,
                height=self.scale_h,
                edgecolor="red",
                facecolor="none",
                linewidth=2
            )
        else:
            patch = patches.Rectangle(
                (xmin, ymin),
                width=self.scale_w,
                height=self.scale_h,
                edgecolor="red",
                facecolor="none",
                linewidth=2
            )

        self.target_overlay = self.target_ax.add_patch(patch)
        self.fig.canvas.draw_idle()

    def setup_widgets(self):
        self.rectangle_selector = RectangleSelector(
            self.source_ax, 
            self.on_source_select, 
            useblit=True,        # Improves performance
            button=[1],          # Left mouse button only
            minspanx=5,          # Minimum width in pixels
            minspany=5,          # Minimum height in pixels
            spancoords='data', # Interpret minspan in pixels
            interactive=True     # Allows moving/reshaping after creation
        )

        self.ellipse_selector = EllipseSelector(
            self.source_ax,
            self.on_source_select, 
            useblit=True,        # Improves performance
            button=[1],          # Left mouse button only
            minspanx=5,          # Minimum width in pixels
            minspany=5,          # Minimum height in pixels
            spancoords='data', # Interpret minspan in pixels
            interactive=True     # Allows moving/reshaping after creation
        )

        self.selector = self.rectangle_selector

        self.rect = patches.Rectangle(
            (1, 1), 
            2, 
            6, 
            edgecolor='red', 
            facecolor='gray', 
            alpha=0.5, 
            linewidth=2
        )

        button_ax = self.fig.add_axes([0.40, 0.3, 0.20, 0.075])
        self.apply_button = Button(
            button_ax,
            "Apply Blend"
        )

        slider_ax = self.fig.add_axes([0.25, 0.10, 0.50, 0.035])
        self.scale_slider = Slider(
            ax=slider_ax,
            label="Scale",
            valmin=0.25,
            valmax=3.0,
            valinit=self.scale,
            valstep=0.05
        )

        shape_ax = self.fig.add_axes([0.03, 0.04, 0.15, 0.13])
        self.shape_buttons = RadioButtons(
            shape_ax,
            ("Rectangle", "Ellipse"),
            active=0
        )

        self.apply_button.on_clicked(self.apply_blend)
        self.scale_slider.on_changed(self.on_scale_change)
        self.shape_buttons.on_clicked(self.on_mask_change)
        self.target_click_cid = self.fig.canvas.mpl_connect(
            "button_press_event", 
            self.on_target_click
        )
        self.zoom_cid = self.fig.canvas.mpl_connect(
            "scroll_event",
            self.on_scroll
        )


    def apply_blend(self, event):
        if self.source_bounds is None or self.target_bounds is None:
            print("Select a source region and destination first.")
            return

        result = self.blender.blend(
            source=self.source_image,
            target=self.target_image,
            source_center=self.source_center,
            target_center=self.target_center,
            target_bounds=self.target_bounds,
            scale=self.scale,
            mask_type=self.mask_type
        )

        self.result_image = result
        self.result_artist.set_data(result)
        self.fig.canvas.draw_idle()

    def run(self):
        if self.fig is None:
            raise RuntimeError("GUI layout has not been initialized.")

        plt.show()


        

