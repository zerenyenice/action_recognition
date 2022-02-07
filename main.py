import shutil

from google_drive_downloader import GoogleDriveDownloader
from core.person_detection import PersonDetection
from core.open_pose import Body, draw_bodypose
from core.action_detection import ActionDetection
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from pathlib import Path


class MainTask:
    def __init__(self):
        self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        self.download_model_data()
        self.person_detector = PersonDetection()
        self.open_pose_detector = Body('model/body_pose_model.pth')
        self.action_detector = ActionDetection('model/torch_state.h5')
        self.frame_list = []
        self.body_points = []
        self.actions = None

    @staticmethod
    def download_model_data():
        GoogleDriveDownloader.download_file_from_google_drive(file_id='1EULkcH_hhSU28qVc1jSJpCh2hGOrzpjK',
                                                              dest_path='./model/body_pose_model.pth', unzip=False)
        #GoogleDriveDownloader.download_file_from_google_drive(file_id='1ZNJDzQUjo2lDPwGoVkRLg77eA57dKUqx',
        #                                                      dest_path='./data/hit_kick_data.zip', unzip=True)

    @staticmethod
    def draw_polygon(image, points, color, thickness, text):
        points_len = len(points)
        for i in range(0, points_len):
            p0 = tuple(points[i])
            p1 = tuple(points[(i + 1) % points_len])
            image = cv2.line(image, p0, p1, color, thickness=thickness)

        image = cv2.putText(image, text, points[3], cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 1, 1)

        return image

    def draw_rect(self,image, rect, color, thickness, text):
        l, t, r, b, _, _ = [int(x) for x in rect]
        image = self.draw_polygon(image, [(l, t), (r, t), (r, b), (l, b)], color, thickness, text)
        return image

    def create_frames_loader(self, video_path):
        return self.person_detector.create_dataset(video_path)

    def detect_person(self,frames):
        for path, im, im0s, vid_cap, s in frames:
            results = self.person_detector.model(im)
            self.frame_list.append(results)
            results.print()

    def detect_body_parts(self):
        imgs = []
        for frame_i in self.frame_list:
            crops = frame_i.crop(save=False)
            for crop_i in crops:
                img = crop_i['im']
                imgs.append(img)
                candidate, subset = self.open_pose_detector(img)
                try:
                    #candidate[:, 0] = candidate[:, 0]
                    #candidate[:, 1] = candidate[:, 1]
                    candidate = np.concatenate([candidate, [[0.0, 0.0, 0.0, 0.0]]])
                    candidate = candidate[subset[0, :-2].astype('int'), 0:2]
                except:
                    candidate = np.zeros((18, 2))
                self.body_points.append(candidate)
                break

        canvas = draw_bodypose(img, candidate[:, :], subset[[0]])
        plt.imshow(canvas[:, :, [2, 1, 0]])

    def get_actions(self):
        self.actions = self.action_detector.recognize_actions(self.body_points)


    def finalize(self):

        if os.path.exists('output'):
            shutil.rmtree('output')
        os.mkdir('output')
        h, w = self.frame_list[0].imgs[0].shape[0:2]
        out_file = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))
        for i, (frame_i, action) in enumerate(zip(self.frame_list, self.actions)):
            img = frame_i.imgs[0][..., ::-1]
            bbox = frame_i.xyxy[0].tolist()[0]
            print(i, bbox, action)
            img = self.draw_rect(img, bbox, self.colors[action], 2, self.action_detector.action_dict[action])
            cv2.imwrite(f'output/{i:05}.jpg', img)
            out_file.write(img)
        out_file.release()


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-i', '--input-video',
        dest='input_video',
        default='test.mp4',
        help='path for input video file')

    options = arg_parser.parse_args()

    main_task = MainTask()
    frame_loader = main_task.create_frames_loader(options.input_video)
    main_task.detect_person(frame_loader)
    print('detecting body joints')
    main_task.detect_body_parts()
    print('recognizing actions')
    main_task.get_actions()
    print('creating output file')
    main_task.finalize()
    print('check -> out.mp4')
