import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

from gui import Application, MainWindow
from camera import Camera, show_frames
from active_learner import build_active_learner
from fovea import Fovea

def active_learning_loop(camera, learner):
    
    fovea = Fovea()
    control = fovea.widget()
    plt.show(block=False)

    keys_to_classes = {}
    classes_to_keys = {}

    while True:
        camera.grab_next_frame()
        frame_views = camera.get_current_views()

        foveated_image = fovea.foveate(frame_views["color"])
        frame_views["color"] = foveated_image.cpu().numpy().astype(int)

        show_frames(frame_views)
    
        # You may need to convert the color.
        color_frame = cv2.cvtColor(frame_views["color"], cv2.COLOR_BGR2RGB)
        color_frame_pil = Image.fromarray(color_frame, mode='RGB')

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            pass
        elif key == -1:
            continue
        else:
            if key not in keys_to_classes:
                if keys_to_classes:
                    c = max(keys_to_classes.values()) + 1
                else:
                    c = 0
                keys_to_classes[key] = c
                classes_to_keys[c] = chr(key)

            if keys_to_classes[key] >= learner.num_classes:
                learner.grow_class_size()
            y = keys_to_classes[key]
            print(f"Changed class to {y}")
            
            y_pred = learner.teach(color_frame_pil, y)

        y_pred = learner(color_frame_pil)
        
        mask_proportion = (
            1 - (fovea.mask.sum(dtype="float32")
                 /fovea.mask.numel())
        ).item()

        print(f"Masked pixel proportion: {mask_proportion: .2}")

        preds = {k: round(y_pred[0, c].item(), 2)
                 for c, k in classes_to_keys.items()}
        print(f"preds: {preds}")
        print(f"predicted key: '{classes_to_keys[y_pred.argmax().item()]}'")

        print("-"*80)

def active_learning_gui(camera, learner, existing_classes=1):
    
    def _process_camera_frame():
        camera.grab_next_frame()
        frame_views = camera.get_current_views()
        show_frames(frame_views)
        
        # You may need to convert the color.
        color_frame = cv2.cvtColor(frame_views["color"], cv2.COLOR_BGR2RGB)
        color_frame_pil = Image.fromarray(color_frame, mode='RGB')
        return color_frame_pil

    def _predict():
        color_frame_pil = _process_camera_frame()

        y_pred = learner(color_frame_pil).cpu().detach().numpy()
        
        w.plot(y_pred[0], clear=True)

        print(f"pred:{y_pred}")
        print(f"pred:{np.argmax(y_pred)}")

    def _train_on_class(class_number):
        color_frame_pil = _process_camera_frame()

        print(f"Training on class {class_number}")
        class_number_pred = learner.teach(color_frame_pil, class_number)
        class_number_pred = class_number_pred.cpu().detach().numpy()

        w.plot(class_number_pred[0], clear=True)

        return class_number_pred

    
    app = Application(sys.argv)
    w = MainWindow(
        existing_classes=existing_classes,
        predict_btn_callback=_predict, 
        class_btn_callback_func=_train_on_class,
        add_class_func=learner.grow_class_size
    )

    w.show()
    sys.exit(app.exec_())

def run_active_learner(existing_classes=1, use_gui=False):
    print("Running ActiveLearner")

    camera = Camera(0, "output")
    learner = build_active_learner(
        num_classes=existing_classes, 
        use_pretrained=False, 
        feature_extract=True, 
        num_epochs=3
    )

    if use_gui:
        active_learning_gui(camera, learner, existing_classes=existing_classes)
    else:
        active_learning_loop(camera, learner)


USE_GUI = True

if __name__ == "__main__":
    run_active_learner(use_gui=USE_GUI)