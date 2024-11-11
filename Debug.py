from time import time

import cv2
import numpy as np

from Mosse_Tracker.TrackerManager import Tracker, TrackerType
from PIL import Image
from Car_Detection_TF.yolo import YOLO
from Car_Detection.detect import Yolo_image, Darknet
from Mosse_Tracker.utils import draw_str
from boxes.yoloFiles import loadFile
from VIF.vif import VIF
vif = VIF()
pi =22/7

tracker_type = TrackerType.MOSSE

def predict(frames_RGB, trackers):
    gray_frames = []
    for frame in frames_RGB:
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    crash = 0

    for tracker in trackers:
        tracker_frames, width, height, xmin, xmax, ymin, ymax = tracker.getFramesOfTracking(gray_frames)

        if tracker_frames == None:
            continue

        if xmax - xmin < 50:
            continue
        if ymax - ymin <= 28:
            continue

        if (ymax- ymin) / (xmax - xmin) <0.35:
            continue

        feature_vec = vif.process(tracker_frames)
        result = vif.clf.predict(feature_vec.reshape(1, 304))
        if result[0] == 0.0:
            pass
        else:
            crash += 1
            tracker.saveTracking(frames_RGB)
    return crash

def checkDistance(frames, tracker_A, tracker_B, frame_no):
    if not tracker_A.isAboveSpeedLimit(frame_no-10, frame_no) and not tracker_B.isAboveSpeedLimit(frame_no-10, frame_no):
        return False

    xa, ya = tracker_A.estimationFutureCenter[frame_no]
    xb, yb = tracker_B.estimationFutureCenter[frame_no]
    r = pow(pow(xa - xb, 2) + pow(ya - yb, 2), 0.5)

    if tracker_type == TrackerType.MOSSE:
        xa_actual, ya_actual = tracker_A.tracker.centers[frame_no]
        xb_actual, yb_actual = tracker_B.tracker.centers[frame_no]
    else:
        xa_actual, ya_actual = tracker_A.get_position(tracker_A.history[frame_no])
        xb_actual, yb_actual = tracker_B.get_position(tracker_B.history[frame_no])
    difference_trackerA_actual_to_estimate = pow(pow(xa_actual - xa, 2) + pow(ya_actual - ya, 2), 0.5)
    difference_trackerB_actual_to_estimate = pow(pow(xb_actual - xb, 2) + pow(yb_actual - yb, 2), 0.5)
    max_difference = max(difference_trackerA_actual_to_estimate, difference_trackerB_actual_to_estimate)

    if r == 0:
        return True

    if r < 40 and max_difference/r > 0.5:
        return True
    return False

def process(trackers, frames):
    new_trackers = trackers
    for i in range(len(new_trackers)):
        for j in range(i+1, len(trackers)):
            if i == j:
                continue
            tracker_A = trackers[i]
            tracker_B = trackers[j]

            if checkDistance(frames, tracker_A, tracker_B, 16) or checkDistance(frames, tracker_A, tracker_B, 19) or checkDistance(frames, tracker_A, tracker_B, 22) or checkDistance(frames, tracker_A, tracker_B, 25) or checkDistance(frames, tracker_A, tracker_B, 28):
                predict(frames, [tracker_B, tracker_A])

class MainFlow:
    def __init__(self, yolo, fromFile=True, select=False):
        self.yolo = yolo
        self.frameCount = 0
        self.readFile = fromFile
        self.selectYOLO = select
        self.trackerId = 0
        self.total_frames = 0
        self.detected_crashes = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        if self.selectYOLO:
            self.model = Darknet("Car_Detection/config/yolov3.cfg", CUDA=False)
            self.model.load_weight("Car_Detection/config/yolov3.weights")
        else:
            self.model = None

    def run(self, path):
        global total_frames
        last_30_frames = []
        last_delayed_30_frames = []
        fileBoxes = []
        new_frame = None
        if self.readFile:
            fileBoxes = loadFile(path)

        cap = cv2.VideoCapture(path)
        frame_width = 480
        frame_height = 360
        trackers = []
        delayed_trackers = []

        fps = 30
        hfps = 15
        paused = False
        cum_time = 0
        while True:
            if not paused:
                t = time()
                ret, frame = cap.read()
                if ret:
                    dim = (480, 360)
                    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                    new_frame = frame.copy()
                    total_frames.append(new_frame)
                else:
                    break

                if self.frameCount > 0 and (self.frameCount % fps == 0 or self.frameCount == fps - 1):
                     t = time()
                     process(trackers, last_30_frames)
                     print(time() - t)

                if self.frameCount > 16 and self.frameCount % hfps == 0 and self.frameCount % fps != 0:
                     t = time()
                     process(delayed_trackers, last_delayed_30_frames)
                     print(time() - t)

                if self.frameCount > 0 and self.frameCount % hfps == 0 and self.frameCount % fps != 0:
                    bboxes = []
                    last_delayed_30_frames = []
                    img = Image.fromarray(frame)

                    if self.readFile:
                        bboxes = fileBoxes[self.frameCount]
                    elif not self.selectYOLO:
                        img, bboxes = self.yolo.detect_image(img)
                    else:
                        bboxes = Yolo_image(np.float32(img), self.model)

                    for i, bbox in enumerate(bboxes):
                        xmin = int(bbox[1])
                        xmax = int(bbox[2])
                        ymin = int(bbox[3])
                        ymax = int(bbox[4])

                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self.trackerId += 1
                        if xmax < frame_width and ymax < frame_height:
                            tr = Tracker(frame_gray, (xmin, ymin, xmax, ymax), frame_width, frame_height,
                                         self.trackerId, tracker_type)
                            delayed_trackers.append(tr)
                        elif xmax < frame_width and ymax >= frame_height:
                            tr = Tracker(frame_gray, (xmin, ymin, xmax, frame_height - 1), frame_width, frame_height,
                                         self.trackerId, tracker_type)
                            delayed_trackers.append(tr)
                        elif xmax >= frame_width and ymax < frame_height:
                            tr = Tracker(frame_gray, (xmin, ymin, frame_width - 1, ymax), frame_width, frame_height,
                                         self.trackerId, tracker_type)
                            delayed_trackers.append(tr)
                        else:
                            tr = Tracker(frame_gray, (xmin, ymin, frame_width - 1, frame_height - 1), frame_width,
                                         frame_height, self.trackerId, tracker_type)
                            delayed_trackers.append(tr)
                else:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    for i, tracker in enumerate(delayed_trackers):
                        left, top, right, bottom = tracker.update(frame_gray)
                        left_future, top_future, right_future, bottom_future = tracker.futureFramePosition()

                if self.frameCount % fps == 0 or self.frameCount == 0:
                    trackers = []
                    bboxes = []
                    last_30_frames = []
                    img = Image.fromarray(frame)

                    if self.readFile:
                        bboxes = fileBoxes[self.frameCount]
                    elif not self.selectYOLO:
                        img, bboxes = self.yolo.detect_image(img)
                    else:
                        bboxes = Yolo_image(np.float32(img), self.model)

                    for i, bbox in enumerate(bboxes):
                        xmin = int(bbox[1])
                        xmax = int(bbox[2])
                        ymin = int(bbox[3])
                        ymax = int(bbox[4])

                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self.trackerId +=1
                        if xmax < frame_width and ymax < frame_height:
                            tr = Tracker(frame_gray, (xmin, ymin, xmax, ymax), frame_width, frame_height,self.trackerId,tracker_type)
                            trackers.append(tr)
                        elif xmax < frame_width and ymax >= frame_height:
                            tr = Tracker(frame_gray, (xmin, ymin, xmax, frame_height - 1), frame_width, frame_height,self.trackerId,tracker_type)
                            trackers.append(tr)
                        elif xmax >= frame_width and ymax < frame_height:
                            tr = Tracker(frame_gray, (xmin, ymin, frame_width - 1, ymax), frame_width, frame_height,self.trackerId,tracker_type)
                            trackers.append(tr)
                        else:
                            tr = Tracker(frame_gray, (xmin, ymin, frame_width - 1, frame_height - 1), frame_width, frame_height,self.trackerId,tracker_type)
                            trackers.append(tr)
                else:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    for i, tracker in enumerate(trackers):
                        left, top, right, bottom = tracker.update(frame_gray)
                        radian = tracker.getCarAngle() * (pi / 180)
                        radian = 0

                        left_future, top_future, right_future, bottom_future = tracker.futureFramePosition()

                        if left > 0 and top > 0 and right < frame_width and bottom < frame_height:
                            if tracker.isAboveSpeedLimit():
                                cv2.rectangle(frame, (int(left), int(top)),(int(right), int(bottom)), (0, 0, 255))
                            else:
                                cv2.rectangle(frame, (int(left), int(top)),(int(right), int(bottom)), (255, 0, 0))

                            draw_str(frame, (left, bottom + 16), 'Avg Speed: %.2f' % tracker.getAvgSpeed())

                        if left_future > 0 and top_future > 0 and right_future < frame_width and bottom_future < frame_height:
                            cv2.rectangle(frame, (int(left_future), int(top_future)), (int(right_future), int(bottom_future)), (0, 255, 0))

                cum_time += time() - t
                cv2.imshow("result", frame)
                last_30_frames.append(new_frame)
                last_delayed_30_frames.append(new_frame)
                if self.frameCount %fps == 0:
                    print(self.frameCount/cum_time)
                self.frameCount += 1
                self.total_frames += 1
                crash = predict(last_30_frames, trackers)
                if crash > 0:
                    self.detected_crashes += 1
                    self.true_positives += 1
            ch = cv2.waitKey(10)
            if ch == ord(' '):
                paused = not paused
        print(self.trackerId)

    def calculate_metrics(self):
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Total frames processed: {self.total_frames}")
        print(f"Detected crashes: {self.detected_crashes}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

if __name__ == '__main__':
    m = MainFlow(None, select=False)
    m.run('videos/1533.mp4')
    m.calculate_metrics()






