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

    fovea_sample = torch.where(fovea_sample <= zero, zero, fovea_sample)
    fovea_sample = torch.where(fovea_sample >= image_shape, image_shape - 1, fovea_sample)

    mask = torch.zeros(image_shape.tolist(), dtype=torch.bool, device=DEVICE)

    mask[fovea_sample[:, 0], fovea_sample[:, 1]] = 1
    return mask


class Fovea:
    def __init__(self, alpha=0.5, fovea_density=0.7, num_samples=50000, 
                 fps=25, device=DEVICE):
        #TODO: Add mean adjustment to allow for dynamic attention
        #TODO: Add standard deviation adjustment to allow for dynamic focus

        self.device = device
        self._fovea_density = torch.tensor(
            fovea_density, dtype=torch.float32, device=self.device)
        self._alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self._num_samples = torch.tensor(num_samples, dtype=torch.int64, device=self.device)
        self.fps = fps
        self.last_frame_time = -1
        self._fovea_width = 32
        self._peripheral_width = 4
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

    @property
    def num_samples(self):
        return self._num_samples.cpu()

    @property
    def fovea_width(self):
        return self._fovea_width

    @property
    def peripheral_width(self):
        return self._peripheral_width

    @fovea_density.setter
    def fovea_density(self, fovea_density):
        self._fovea_density = torch.tensor(
            fovea_density, dtype=torch.float64, device=self.device)

    @num_samples.setter
    def num_samples(self, num_samples):
        self._num_samples = torch.tensor(
            num_samples, dtype=torch.int32, device=self.device)

    @fovea_width.setter
    def fovea_width(self, fovea_width):
        self._fovea_width = torch.tensor(
            fovea_width, dtype=torch.float64, device=self.device)

    @peripheral_width.setter
    def peripheral_width(self, peripheral_width):
        self._peripheral_width = torch.tensor(
            peripheral_width, dtype=torch.int32, device=self.device)


    def generate_mask(self, image_shape=DEFAULT_MASK_SHAPE):
        sample = generate_fovea_sample(
            self._num_samples, 
            self._fovea_density, 
            image_shape=image_shape,
            fovea_width_factor=self._fovea_width, 
            spread_width_factor=self._peripheral_width)
        self.mask = generate_mask_from_fovea_sample(sample, image_shape)
        return self.mask

    def update_mask_view(self):
        # update the image
        self.image.set_data(self.mask.cpu())

    def widget(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        self.fig = plt.figure()

        self.ax_image            = plt.axes([0.00, 0.25, 1.00, 0.75])
        self.ax_fovea_density    = plt.axes([0.25, 0.16, 0.50, 0.04])
        self.ax_num_samples      = plt.axes([0.25, 0.11, 0.50, 0.04])
        self.ax_fovea_width      = plt.axes([0.25, 0.06, 0.50, 0.04])
        self.ax_peripheral_width = plt.axes([0.25, 0.01, 0.50, 0.04])

        self.image = self.ax_image.imshow(self.mask.cpu())

        # Setup sliders
        self.slider_fovea_density = Slider(
            ax=self.ax_fovea_density, 
            label='Fovea Density', 
            valmin=0, 
            valmax=1, 
            valinit=self.fovea_density
        )
        self.slider_num_samples = Slider(
            ax=self.ax_num_samples, 
            label='Num Samples', 
            valmin=0, 
            valmax=self.num_samples*3, 
            valinit=self.num_samples, 
            valfmt='%1d'
        )
        self.slider_fovea_width = Slider(
            ax=self.ax_fovea_width, 
            label='Fovea Width', 
            valmin=4, 
            valmax=128, 
            valinit=self.fovea_width
        )
        self.slider_peripheral_width = Slider(
            ax=self.ax_peripheral_width, 
            label='Peripheral Width', 
            valmin=2, 
            valmax=64, 
            valinit=self.peripheral_width
        )

        # Add callback function to change associated member valiables
        # when sliders are moved
        self.slider_fovea_density.on_changed(
            lambda val: setattr(self, "fovea_density", val))
        self.slider_num_samples.on_changed(
            lambda val: setattr(self, "num_samples", val))
        self.slider_fovea_width.on_changed(
            lambda val: setattr(self, "fovea_width", val))
        self.slider_peripheral_width.on_changed(
            lambda val: setattr(self, "peripheral_width", val))
        
        self.slider_fovea_density.on_changed(
            lambda val: self.update_mask_view())
        self.slider_num_samples.on_changed(
            lambda val: self.update_mask_view())
        self.slider_fovea_width.on_changed(
            lambda val: self.update_mask_view())
        self.slider_peripheral_width.on_changed(
            lambda val: self.update_mask_view())

        plt.show(block=False)


def camera_fovea_demo():
    import cv2
    from camera import Camera, show_frame
    
    camera = Camera(0, "output", fps=45)

    fovea = Fovea(fps=5)
    control = fovea.widget()
    while camera.grab_next_frame():
        foveated_image = fovea.foveate(camera.frames[-1])

        show_frame(frame_name="color", 
                   frame=foveated_image.cpu().type(torch.uint8).numpy())

        # process key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == -1:
            continue

        print(fovea.mask.sum(dtype=torch.float32)/fovea.mask.numel())


if __name__ == "__main__":
    camera_fovea_demo()

