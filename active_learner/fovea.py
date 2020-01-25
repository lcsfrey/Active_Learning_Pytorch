import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
import torch.distributions as td

DEVICE_TYPE = "cuda:0" if torch.cuda.is_available() else "cpu:0"
DEVICE = torch.device(DEVICE_TYPE)

def generate_fovea_sample(num_samples, fovea_density, image_shape, 
                           fovea_width_factor=16, spread_width_factor=4):
    image_shape_tensor = torch.tensor(image_shape, dtype=torch.int64, device=DEVICE)
    center_dist = td.normal.Normal(
        image_shape_tensor.type(torch.float32) / 2, 
        image_shape_tensor.type(torch.float32) / fovea_width_factor, 
        validate_args=None)

    num_fovera_samples = (num_samples * fovea_density).type(torch.int64)
    center = center_dist.sample([num_fovera_samples]).round_()
    num_spread_samples = num_samples - num_fovera_samples

    spread_dist = td.normal.Normal(
        image_shape_tensor.type(torch.float32) / 2, 
        image_shape_tensor.type(torch.float32) / spread_width_factor, 
        validate_args=None)
    spread = spread_dist.sample([num_spread_samples]).round_()

    fovea = torch.cat([center, spread], dim=0).type(torch.int64)

    return fovea

def generate_mask_from_sample(sample, image_shape):
    image_shape_tensor = torch.tensor(image_shape, dtype=torch.int64, device=DEVICE)
    zero = torch.zeros(1, dtype=torch.int64, device=DEVICE)

    sample = torch.where(sample < zero, zero, sample)
    sample = torch.where(sample >= image_shape_tensor, image_shape_tensor - 1, sample)

    mask = torch.zeros(image_shape.tolist())

    mask[sample[:, 0], sample[:, 1]] = 1
    return mask

def fovea_density_demo():    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider
    import time
        

    # Generate some data for this demonstration.
    image_shape = [1080, 1920]
    alpha = 0.5
    num_samples = (alpha * torch.prod(torch.tensor(image_shape, device=DEVICE))).type(torch.int64).cuda()
    fovea_density = .3

    sample = generate_fovea_sample(num_samples, fovea_density, image_shape)
    mask = generate_mask_from_sample(sample, image_shape)

    fig, ax = plt.subplots()
    image = plt.imshow(mask.cpu())

    ax_fovea = plt.axes([0.25, .03, 0.50, 0.02])
    ax_num_samples = plt.axes([0.25, .08, 0.50, 0.02])

    # Slider
    slider_fovea = Slider(ax_fovea, 'Fovea Density', 0, 1, valinit=fovea_density)
    slider_num_samples = Slider(ax_num_samples, 'Num Samples', 0, num_samples.cpu()*3, valinit=num_samples.cpu(), valfmt='%1d')
    
    def update_fovea(val):
        nonlocal fovea_density
        # amp is the current value of the slider
        fovea_density = torch.tensor(slider_fovea.val, dtype=torch.float64, device=DEVICE)
        update_mask()

    def update_num_samples(val):
        nonlocal num_samples
        num_samples = torch.tensor(slider_num_samples.val, dtype=torch.int64, device=DEVICE)
        update_mask()

    def update_mask():
        sample = generate_fovea_sample(num_samples, fovea_density, image_shape)
        mask = generate_mask_from_sample(sample, image_shape)

        # update the image
        image.set_data(mask.cpu())

        # redraw canvas while idle
        fig.canvas.draw_idle()

    # call update function on slider value change
    slider_fovea.on_changed(update_fovea)
    slider_num_samples.on_changed(update_num_samples)

    plt.show()





class Fovea:
    def __init__(self, image_shape, alpha=0.5, fovea_density=.3, device=DEVICE):
        self.device = device
        self._fovea_density = torch.tensor(
            fovea_density, dtype=torch.float64, device=self.device)
        self.image_shape_tensor = torch.tensor(
            image_shape, dtype=torch.int64, device=self.device)
        self._num_samples = (alpha * self.image_shape_tensor.prod()).type(torch.int64)

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
            num_samples, dtype=torch.int64, device=self.device)

    def generate_mask(self):
        sample = generate_fovea_sample(
            self.num_samples, self.fovea_density, self.image_shape_tensor)
        self._mask = generate_mask_from_sample(sample, self.image_shape_tensor)
        return self._mask

    def update_mask(self):
        # update the image
        self.image.set_data(self.generate_mask().cpu())

        # redraw canvas while idle
        self.fig.canvas.draw_idle()

    def setup_fovea_control(self):
        from matplotlib.widgets import Slider
        self.fig, self.ax = plt.subplots()
        self.image = plt.imshow(self.generate_mask().cpu())

        self.ax_fovea = plt.axes([0.25, .03, 0.50, 0.02])
        self.ax_num_samples = plt.axes([0.25, .08, 0.50, 0.02])

        # Slider
        self.slider_fovea = Slider(self.ax_fovea, 'Fovea Density', 0, 1, valinit=self.fovea_density)
        self.slider_num_samples = Slider(self.ax_num_samples, 'Num Samples', 0, self.num_samples*3, 
                                    valinit=self.num_samples, valfmt='%1d')
        

        # call update function on slider value change
        self.slider_fovea.on_changed(lambda val: self.update_fovea_density(val))
        self.slider_num_samples.on_changed(lambda val: self.update_num_samples(val))
        
        self.slider_fovea.on_changed(lambda val: self.update_mask())
        self.slider_num_samples.on_changed(lambda val: self.update_mask())



def camera_fovea_demo():

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider
    import time

    import cv2
    from PIL import Image
    from camera import Camera, show_frame
    
    camera = Camera(0, "output")
    # grab the next frame
    camera.grab_next_frame()
    # get all views of the current camera frame
    frame_views = camera.get_current_views()
    
    image_shape = frame_views["color"].shape[:2]

    fovea = Fovea(image_shape)
    control = fovea.setup_fovea_control()
    plt.show(block=False)

    while True:
        camera.grab_next_frame()
        frame_views = camera.get_current_views()

        mask = fovea.generate_mask()
        frame_views["color"] = np.where(mask[..., None].cpu().numpy(), frame_views["color"], 0)

        show_frame(frame_views)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == -1:
            camera.processKey(key)
            continue

if __name__ == "__main__":
    camera_fovea_demo()

