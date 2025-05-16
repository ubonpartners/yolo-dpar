import os
import argparse
import cv2
import gdown
import stuff
import ultralytics

def get_weights(model):
    """
    Check if weights .pt file exists and if not download it from gdrive
    """
    weight_ids={"yolo-dpa-l":"1DwRpgS53MtQYM4G7Rm1K7OBxHhguaiI5",
                "yolo-dpa-s":"1FUK6x26Z8Dz0gqw-20IHrvnUIKl8lLhk",
                "yolo-dpa-n":"1YDbFnwfd_xvlm4kkRiXCs_FMCPPOTfXP",
                "yolo-dp-l":"1veVJ9y6Set3oIDtZ47_Zpz6cnYqyMauy"}

    assert model in weight_ids, "must specify weights in the table"

    weight_filename=model+".pt"

    if not os.path.exists(weight_filename):
        url = f"https://drive.google.com/uc?id={weight_ids[model]}"
        print(f"Downloading {weight_filename} from gdrive url {url}")
        gdown.download(url, weight_filename, quiet=False)

    assert os.path.exists(weight_filename), "No weights!"
    return weight_filename

def do_video(video, model):

    yolo=ultralytics.YOLO(model)
    class_names=[yolo.names[i] for i in range(len(yolo.names))]

    if video=="webcam":
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(),  "Error: Cannot access the webcam"
        width=1280
        height=720
        fps=30
        frame_count=0
        # Step 2: Set video properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set frame width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Set frame height
        cap.set(cv2.CAP_PROP_FPS, fps)  # Set FPS (frames per second)
        print("Using video from webcam")
    else:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Error: Cannot open the video file {video}")
            exit()

        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
        print(f"Video {width}x{height} {fps}fps {frame_count} frames")

    attributes=[]
    for c in class_names:
        if c.startswith("person_"):
            attributes.append("person:"+c[len("person_"):])

    paused=False

    display_width=1600
    display_height=900
    highlight_pos=None
    
    display=stuff.Display(width=display_width, height=display_height)

    while True:
        if paused is False:
            ret, frame = cap.read()  # Read a frame from the video
            
        result=yolo(frame, conf=0.2, max_det=500, half=True, verbose=False)

        if not ret:
            break  # Break if no frame is read (end of video)

        out_det=stuff.yolo_results_to_dets(result[0],
                                         det_thr=0.2,
                                         yolo_class_names=class_names,
                                         class_names=class_names,
                                         attributes=attributes,
                                         face_kp=True,
                                         pose_kp=True,
                                         fold_attributes=True)
        display.clear()

        highlight_index=None
        if highlight_pos is not None:
             highlight_index, dist1=stuff.find_gt_from_point(out_det, highlight_pos[0], highlight_pos[1])
        stuff.draw_boxes(display,
                         out_det,
                         attributes=attributes,
                         highlight_index=highlight_index,
                         class_names=class_names)
    
        display.show(frame, title="results")
        events=display.get_events(5)
        for e in events:
            if e['key']=='p':
                paused=not paused
            if e['lbutton']:
                highlight_pos=[e['x'], e['y']]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='view.py')
    parser.add_argument('--model', type=str, default='yolo-dpa-l', help='model to use')
    parser.add_argument('--video', type=str, default='webcam', help='video source; webcam or path to mp4')
    opt = parser.parse_args()

    weights=get_weights(opt.model)
    do_video(opt.video, weights)