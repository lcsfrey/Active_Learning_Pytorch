import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
import torch.distributions as td
import time

DEVICE_TYPE = "cuda:0" if torch.cuda.is_available() else "cpu:0"
DEVICE = torch.device(DEVICE_TYPE)
DEFAULT_MASK_SHAPE = torch.tensor((400,400), dtype=torch.int64, device=DEVICE)

def generate_fovea_sample(num_samples, fovea_density, image_shape, 
                           fovea_width_factor=32, spread_width_factor=4):

    float_image_shape = torch.tensor(image_shape, dtype=torch.float32, device=DEVICE)
    center_dist = td.normal.Normal(float_image_shape / 2, 
                                   float_image_shape / fovea_width_factor, 
                                   validate_args=None)

    num_fovera_samples = (num_samples * fovea_density).type(torch.int64)
    center = center_dist.sample([num_fovera_samples]).round_()
    num_spread_samples = num_samples - num_fovera_samples

    spread_dist = td.normal.Normal(float_image_shape / 2, 
                                  float_image_shape / spread_width_factor, 
                                  validate_args=None)
    spread = spread_dist.sample([num_spread_samples]).round_()

    fovea_sample = torch.cat([center, spread], dim=0).type(torch.int64)

    return fovea_sample

def generate_mask_from_fovea_sample(fovea_sample, image_shape):
    image_shape = torch.tensor(image_shape, dtype=torch.int64, device=DEVICE)
    zero = torch.zeros(1, dtype=torch.int64, device=DEVICE)

    fovea_sample = torch.where(fovea_sample < zero, zero, fovea_sample)
    fovea_sample = torch.where(fovea_sample >= image_shape, image_shape - 1, fovea_sample)

    mask = torch.zeros(image_shape.tolist(), dtype=torch.bool, device=DEVICE)

    mask[fovea_sample[:, 0], fovea_sample[:, 1]] = 1
    return mask


class Fovea:
    def __init__(self, alpha=0.5, fovea_density=0.7, num_samples=50000, fps=25, device=DEVICE):
        #TODO: Add mean adjustment to allow for dynamic attention
        #TODO: Add standard deviation adjustment to allow for dynamic focus

        self.device = device
        self._fovea_density = torch.tensor(
            fovea_density, dtype=torch.float32, device=self.device)
        self._alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self._num_samples = torch.tensor(num_samples, dtype=torch.int64, device=self.device)
        self.fps = fps
        self.last_frame_time = -1
        self.mask = self.generate_mask()

    def foveate(self, image):
        t = time.time()
        image = torch.tensor(image, dtype=torch.int64, device=DEVICE)
        if 1./self.fps - (t - self.last_frame_time) < 0:
            self.last_frame_time = time.time()
            self.mask = self.generate_mask(image.shape[:2])
        foveated_image = self.apply_mask(image, self.mask)
        
        return foveated_image

    def apply_mask(self, image, mask):
        # TODO: Convert masked pixels from RGB to grayscale.
        #       This approach would more closely mimic the behaviour
        #       of rod cells your peripheral vision.
        if mask.dim() == image.dim() - 1:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, image, torch.zeros_like(image))

    @property
    def fovea_density(self):
        return self._fovea_density.cpu()

    def update_fovea_density(self, fovea_density):
        # amp is the current value of the slider
        self._fovea_density = torch.tensor(
            fovea_density, dtype=torch.float64, device=self.device)

    @property
    def num_samples(self):
        return self._num_samples.cpu()

    def update_num_samples(self, num_samples):
        self._num_samples = torch.tensor(
            num_samples, dtype=torch.int32, device=self.device)

    def generate_mask(self, image_shape=DEFAULT_MASK_SHAPE):
        sample = generate_fovea_sample(
            self.num_samples, self.fovea_density, image_shape)
        self.mask = generate_mask_from_fovea_sample(sample, image_shape)
        return self.mask

    def update_mask_view(self):
        # update the image
        self.image.set_data(self.mask.cpu())

        # redraw canvas while idle
        self.fig.canvas.draw_idle()

    def widget(self):
        from matplotlib.widgets import Slider
        self.fig, self.ax = plt.subplots()
        self.image = plt.imshow(self.mask.cpu())

        self.ax_fovea = plt.axes([0.25, .03, 0.50, 0.02])
        self.ax_num_samples = plt.axes([0.25, .08, 0.50, 0.02])

        # Fovea slider
        num_samples = self._num_samples.cpu()
        self.slider_fovea = Slider(self.ax_fovea, 'Fovea Density', 0, 1, valinit=self.fovea_density)
        self.slider_num_samples = Slider(self.ax_num_samples, 'Num Samples', 0, num_samples*3, 
                                    valinit=num_samples, valfmt='%1d')
        

        # call update function on slider value change
        self.slider_fovea.on_changed(lambda val: self.update_fovea_density(val))
        self.slider_num_samples.on_changed(lambda val: self.update_num_samples(val))
        
        self.slider_fovea.on_changed(lambda val: self.update_mask_view())
        self.slider_num_samples.on_changed(lambda val: self.update_mask_view())



def camera_fovea_demo():

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider

    import cv2
    from PIL import Image
    from camera import Camera, show_frame
    
    camera = Camera(0, "output", fps=45)

    fovea = Fovea(fps=5)
    control = fovea.widget()
    plt.show(block=False)
    while True:
        camera.grab_next_frame()
        frame_views = camera.get_current_views()

        foveated_image = fovea.foveate(frame_views["color"])
        frame_views["color"] = foveated_image.cpu().type(torch.uint8).numpy()

        show_frame(frame_views)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == -1:
            camera.processKey(key)
            continue
        else:
            print(fovea.mask.sum(dtype=torch.float32)/fovea.mask.numel())


if __name__ == "__main__":
    camera_fovea_demo()

