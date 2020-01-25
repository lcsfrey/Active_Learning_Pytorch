import cv2
import numpy as np
import time
from collections import deque

def showFrame(frame_name, frame):
    cv2.imshow(frame_name, frame)

def show_frame(frames: dict):
    for frame_name, frame in frames.items():
        showFrame(frame_name, frame)


# Webcam object used to interact with frames
class Camera:

    # Initializes camera and Mats
    def __init__(self, device_number, output_file, fps=25, previous_frames_stored=5, show=True):
        self.outfilename = output_file
        self._initialize_camera(device_number)

        self._frames = deque(
            [self.cap.read()[1] for _ in range(previous_frames_stored)]
        )
        self.max_frames = previous_frames_stored
        self.frame_count = 0
        self.last_frame_time = -1
        self.fps = fps

        if show:
            self.frame_view_functions = {
                "color":self.get_next_frame, 
                #"threshold": self.get_threshold_frame,
                #"canny": self.get_canny_frame
            }
            for frame_name in self.frame_view_functions.keys():
                cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
            
    def _initialize_camera(self, device_number):
        self.cap = cv2.VideoCapture(device_number)

    def widget(self):            
        # frame conversion properties
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.cannyMin = 30
        self.cannyMax = 150
        self.threshMin = 40
        self.threshMax = 220
        self.minPixels = 500

        # Create a black image, a window
        cv2.namedWindow('Image Controls')

        # create trackbars for thresholds
        cv2.createTrackbar('cannyMin', 'Image Controls', self.cannyMin, 255, self.process_trackbars)
        cv2.createTrackbar('cannyMax', 'Image Controls', self.cannyMax, 255, self.process_trackbars)
        cv2.createTrackbar('threshMin', 'Image Controls', self.threshMin, 255, self.process_trackbars)
        cv2.createTrackbar('threshMax', 'Image Controls', self.threshMax, 255, self.process_trackbars)

    @property
    def frames(self):
        return self._frames

    def get_current_views(self):
        t = time.time()
        if 1./self.fps - (t - self.last_frame_time) < 0:
            self.last_frame_time = time.time()
            self.views = {view_name: view() for view_name, view in self.frame_view_functions.items()}
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
        canny = cv2.Canny(self.get_gray_frame(), self.cannyMin, self.cannyMax)
        for i in range (dilation_iterations):
            canny = cv2.dilate(canny, dilation_kernel)
        return canny

    def get_threshold_frame(self, image_1=None, image_2=None, min_thresh=None, max_thresh=None):
        t_gray1 = cv2.cvtColor(self._frames[-1].copy(), cv2.COLOR_BGR2GRAY)
        t_gray2 = cv2.cvtColor(self._frames[-2].copy(), cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(t_gray1, t_gray2)
        ret, threshold = cv2.threshold(difference, self.threshMin, self.threshMax, cv2.THRESH_BINARY)
        threshold = cv2.dilate(threshold, (3, 3))
        return threshold

    # get current positions of trackbars
    def process_trackbars(self, x):
        print(x)
        self.cannyMin = cv2.getTrackbarPos('cannyMin', 'Image Controls')
        self.cannyMax = cv2.getTrackbarPos('cannyMax', 'Image Controls')
        self.threshMin = cv2.getTrackbarPos('threshMin', 'Image Controls')
        self.threshMax = cv2.getTrackbarPos('threshMax', 'Image Controls')

    def display_text(self, image, text, color=(0, 0, 255)):
        cv2.putText(image, text, (5, self.text_display_offset), self.FONT, .75, color, 2)
        self.text_display_offset += 20
        
    # update frames
    def grab_next_frame(self):
        ret, new_frame = self.cap.read()
        assert ret
        self._frames.append(new_frame)
        old_frame = self._frames.popleft()

        self.frame_count += 1
        self.text_display_offset = 20

    def processKey(self, key):
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

def main():
    print("Camera Demo")
    camera = Camera(0, "output")
    widget = camera.widget()
    key = ''

    class_names = []

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        else:
            camera.processKey(key)

        # grab the next frame
        camera.grab_next_frame()
        
        # get all views of the current camera frame
        frame_views = camera.get_current_views()

        show_frame(frame_views)

if __name__ == "__main__":
    main()
