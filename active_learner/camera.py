import cv2
import numpy as np
import time
from collections import deque

def show_frame(frame_name, frame):
    cv2.imshow(frame_name, frame)

def show_frames(frames: dict):
    for frame_name, frame in frames.items():
        show_frame(frame_name, frame)


# Webcam object used to interact with frames
class Camera:

    # Initializes camera and Mats
    def __init__(self, device_number, output_file, fps=25, 
                 previous_frames_stored=5, show=True, view_names=["color"]):
        self.outfilename = output_file
        self._initialize_camera(device_number)

        self._frames = deque(
            [self.cap.read()[1] for _ in range(previous_frames_stored)]
        )
        self.max_frames = previous_frames_stored
        self.frame_count = 0
        self.last_frame_time = -1
        self.fps = fps
        self.view_names = view_names

        if show:
            frame_view_functions = {
                "color":self.get_next_frame, 
                "threshold": self.get_threshold_frame,
                "canny": self.get_canny_frame
            }
            self.frame_view_functions = {
                view_name: frame_view_functions[view_name] 
                for view_name in self.view_names
            }
            for frame_name in self.frame_view_functions.keys():
                cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
            
    def _initialize_camera(self, device_number):
        self.cap = cv2.VideoCapture(device_number)

    def widget(self):            
        # frame conversion properties
        self.canny_min = 30
        self.canny_max = 150
        self.thresh_min = 40
        self.thresh_max = 220

        # Create a black image, a window
        cv2.namedWindow('Image Controls')

        # create trackbars for thresholds
        cv2.createTrackbar(
            'canny_min', 
            'Image Controls', 
            self.canny_min, 
            255, 
            lambda val: setattr(self, 'canny_min', val)
        )
        cv2.createTrackbar(
            'canny_max', 
            'Image Controls', 
            self.canny_max, 
            255, 
            lambda val: setattr(self, 'canny_max', val)
        )
        cv2.createTrackbar(
            'thresh_min', 
            'Image Controls', 
            self.thresh_min, 
            255, 
            lambda val: setattr(self, 'thresh_min', val)
        )
        cv2.createTrackbar(
            'thresh_max', 
            'Image Controls', 
            self.thresh_max, 
            255, 
            lambda val: setattr(self, 'thresh_max', val)
        )

    def __getitem__(self, index):
        return self._frames[index]

    @property
    def frames(self):
        return self._frames

    def get_current_views(self):
        t = time.time()
        if 1./self.fps - (t - self.last_frame_time) < 0:
            self.last_frame_time = time.time()
            self.views = {
                view_name: view() 
                for view_name, view in self.frame_view_functions.items()
            }
        return self.views

    # Process frame and output the result
    # input:  string indicating what image to get
    #         image to invert if choice is "inverted"
    #
    # output: outputs Mat of your choice
    def get_next_frame(self):
        return self._frames[-1]
    
    def get_gray_frame(self, image=None):
        grey_blurred = None
        if image is None:
            grey = cv2.cvtColor(self._frames[-1].copy(), cv2.COLOR_BGR2GRAY)
            grey_blurred = cv2.blur(grey, (3,3))
        else:
            grey_blurred = cv2.blur(image.copy(), (3,3))
        return grey_blurred

    def get_canny_frame(self, image=None, dilation_kernel=(3,3), dilation_iterations=1, 
                    canny_min=None, canny_max = None):
        canny = cv2.Canny(self.get_gray_frame(), self.canny_min, self.canny_max)
        for i in range (dilation_iterations):
            canny = cv2.dilate(canny, dilation_kernel)
        return canny

    def get_threshold_frame(self, image_1=None, image_2=None, min_thresh=None, max_thresh=None):
        t_gray1 = cv2.cvtColor(self._frames[-1].copy(), cv2.COLOR_BGR2GRAY)
        t_gray2 = cv2.cvtColor(self._frames[-2].copy(), cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(t_gray1, t_gray2)
        ret, threshold = cv2.threshold(difference, self.thresh_min, self.thresh_max, cv2.THRESH_BINARY)
        threshold = cv2.dilate(threshold, (3, 3))
        return threshold
        
    # update frames
    def grab_next_frame(self):
        old_frame = self._frames.popleft()
        ret, new_frame = self.cap.read(image=old_frame)
        self._frames.append(new_frame)

        self.frame_count += 1
        return ret

    def process_key(self, key):
        if key == ord('q'):
            return 'q'


class MyVideoWriter:
    def __init__(self, device_number, output_file):
        self.out = None
        self.can_record = true

    # Read in number to use for new file name
    # Initialize recording
    # returns a file writing object
    def start_recording(self):
        myfile = open('output_count.txt', 'r+')
        filecount = -1
        filecount = myfile.read()
        myfile.close()
        int_filecount = int(filecount)
        if int_filecount is -1:
            int_filecount = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        str_filecount = str(int_filecount)
        filename = 'movement_#' + str_filecount
        mytime = time.strftime("  %d_%m_%Y")
        filename = filename + mytime + '.avi'
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        int_filecount = int_filecount + 1
        myfile = open('output_count.txt', 'w')
        str_filecount = str(int_filecount)
        myfile.write(str_filecount)
        myfile.close()
        return out

    def camera_intialize(self):
        if self.can_record is True:
            self.out = start_recording()

def camera_demo():
    print("Camera Demo")
    camera = Camera(0, "output", view_names=["color", "threshold", "canny"])
    widget = camera.widget()

    while camera.grab_next_frame():
        # get all views of the current camera frame
        frame_views = camera.get_current_views()

        show_frames(frame_views)

        # process key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        else:
            camera.process_key(key)

if __name__ == "__main__":
    camera_demo()
