from threading import Timer
from time import sleep

import cv2
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from ultralytics import YOLO

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def normaly(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    # img = img.reshape(heights, widths, 3)
    img = img / 255
    return img


def relu(x):
    return x * (x > 0)


# video link: https://drive.google.com/file/d/16PClrGCJGV-aSrjevYHA_htyzg2QC49x/view?usp=sharing
cap = cv2.VideoCapture("Germany_480p.mp4")  # from video
# cap = cv2.VideoCapture(0)  #from camera
model1 = YOLO("Manual yolov8n2.pt")  # load detection model
model2 = load_model("manual LeNet4 (no pre).h5")  # load classification model

ano = [
    "Do not enter",
    "No stopping or parking",
    "No parking",
    "Maximum speed 40 km/h",
    "Maximum speed 50 km/h",
    "Maximum speed 60 km/h",
    "Maximum speed 70 km/h",
    "Maximum speed 80 km/h",
    "No right turn",
    "No left turn",
    "Keep right",
    "No 2 & 3 wheel vehicles",
    "Roundabout ahead",
    "No cars",
    "No U-turn",
    "No buses",
    "Bus stop",
    "No motorcycles",
    "Height restriction",
    "No heavy vehicles",
    "Left road junction with priority",
    "Right road junction with priority",
    "Zebra crossing / crosswalk ahead",
    "Traffic obstruction ahead - may pass on either side",
    "Slow down",
    "School zone ahead",
    "Road narrows ahead on the left side",
    "No trailers",
    "Hospital nearby",
    "null",
]


print("model loaded")

# create a dictionary of all trackers in OpenCV that can be used for tracking
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create,
}


# Create MultiTracker object
trackers = cv2.legacy.MultiTracker_create()
label = np.array([], dtype=int)
fnum = 0  # frame .no
avgfps = 0
sampling_rate = 1

frame = cap.read()[1]
fnum += 1

current_frame = frame

frame_height, frame_width = frame.shape[:2]

size = (frame_width, frame_height)

result = cv2.VideoWriter(
    "detection_tracking_result.avi", cv2.VideoWriter_fourcc(*"MJPG"), 60, size
)

is_done = False


def update_writer_task(done):
    if done:
        return
    result.write(current_frame)
    capture_timer = Timer(float(1) / 60, update_writer_task, args=[is_done])
    capture_timer.start()


capture_timer = Timer(float(1) / 60, update_writer_task, args=[is_done])
capture_timer.start()

sleep(0.1)

while True:
    timer = cv2.getTickCount()  # for fps caculating
    frame = cap.read()[1]
    fnum += 1
    boxes = []

    if (fnum % sampling_rate) == 0:
        # run detection model
        results = model1(frame, verbose=False)
        results1 = results[0].boxes
        ima = np.array(frame)
        # create new multi tracker
        trackers = cv2.legacy.MultiTracker_create()
        # clear existing label list
        label = np.array([], dtype=object)
        # get result from detection model
        ims = []
        rois = []
        for bo in results1:
            row = bo.xyxy[0].type(torch.int64).tolist()
            con = round(float(bo.conf[0]), 2)
            # if detection confident lower than 80, skip
            if con < 0.3:
                continue
            xmin = relu(row[0] - 5)
            ymin = relu(row[1] - 5)
            xmax = relu(row[2] + 5)
            ymax = relu(row[3] + 5)

            width = row[2] - row[0]
            height = row[3] - row[1]

            # add bounding box to tracker
            roi = [row[0], row[1], row[2] - row[0], row[3] - row[1]]
            # preprocess image
            cropped_image = ima[ymin:ymax, xmin:xmax]
            im = normaly(cropped_image)

            rois += [roi]
            ims += [im]

        boxes = rois
        if len(ims) > 0:
            y_preds = model2.predict(np.stack(ims, axis=0), verbose=0)
            cls_preds = y_preds.argmax(-1)

            # for roi in rois:
            #     tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
            #     trackers.add(tracker, frame, tuple(roi))

            label = cls_preds

    if frame is None:
        break

    # (success, boxes) = trackers.update(frame)
    # # loop over the bounding boxes and draw then on the frame
    # if success == False:
    #     bound_boxes = trackers.getObjects()
    #     idx = np.where(bound_boxes.sum(axis=1) != 0)[0]
    #     bound_boxes = bound_boxes[idx]
    #     trackers = cv2.legacy.MultiTracker_create()
    #     for bound_box in bound_boxes:
    #         trackers.add(tracker, frame, bound_box)
    #     continue

    # put bounding box
    for i, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            str(ano[label[i]]),
            (x + 10, y - 3),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 255, 0),
            2,
        )
    k = cv2.waitKey(1)
    # caculate fps
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if fnum == 1:
        avgfps = fps
    else:
        avgfps = 0.9 * avgfps + 0.1 * fps

    cv2.putText(
        frame,
        "fps: {:4d}".format(int(avgfps)),
        (75, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    current_frame = frame

    if k == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break

    # sleep(0.01)

is_done = True
capture_timer.cancel()
result.release()
cap.release()
cv2.destroyAllWindows()
