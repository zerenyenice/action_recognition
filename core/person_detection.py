import os
from pathlib import Path
import cv2
import numpy as np
import torch


class Frames:
    VID_FORMATS = ['mp4', 'mov', 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mpeg', 'mpg', 'wmv']

    def __init__(self, video_input, img_size=640, stride=32, auto=True):

        p = str(Path(video_input).resolve())
        if os.path.isfile(video_input):
            files = [video_input]
        else:
            raise Exception(f'ERROR: {p} not found')

        self.files = [x for x in files if x.split('.')[-1].lower() in self.VID_FORMATS]

        self.nf = len(files)
        self.stride = stride
        self.new_video(files[0])
        self.auto = auto
        self.img_size = img_size

    def new_video(self, path):
        self.frame = 0
        self.capture = cv2.VideoCapture(path)
        self.frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.n_f

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        ret_val, img0 = self.capture.read()
        while not ret_val:
            self.count += 1
            self.capture.release()
            if self.count == self.nf:  # last video
                raise StopIteration
            else:
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()

        self.frame += 1
        s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}:'

        # Padded resize
        img = self.letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.capture, s

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


class PersonDetection():
    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.classes = [0]
        self.model = model


    @staticmethod
    def create_dataset(video_file):
        video_file = str(video_file)
        imgsz = (640, 640)
        stride = 32

        data = Frames(video_file, img_size=imgsz, stride=stride, auto=True)

        return data
