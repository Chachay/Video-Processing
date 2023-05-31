import h5py
import argparse
import numpy as np

import cv2
from tqdm import tqdm

from decord import VideoReader
from decord import cpu

from insightface_func.face_detect_crop_multi import Face_detect_crop

def video2hdf5(path, db):
    crop_size = 224

    #cap = cv2.VideoCapture(path)
    vr = VideoReader(path, ctx=cpu(0))

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640, 640),mode='None')

    with h5py.File(db, "w") as h5:
        images_group = h5.create_group("images")
        frame_group = images_group.create_group("full")
        cropped_group = images_group.create_group("cropped")

        matrix_group = h5.create_group("matrix")

        for i in tqdm(range(len(vr))):
            #ret, frame = cap.read()
            frame = vr[i].asnumpy()

            image_dataset = frame_group.create_dataset(
                name=f'{i:08}', data=frame, compression="gzip"
            )

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cropped_img, mat = app.get(frame_bgr, crop_size)
            if len(cropped_img) > 0:
                face = cv2.cvtColor(cropped_img[0], cv2.COLOR_BGR2RGB)
                image_dataset = cropped_group.create_dataset(
                    name=f'{i:08}', data=face, compression="gzip"
                )
                matrix_dataset = matrix_group.create_dataset(
                    name=f'{i:08}', data=mat[0], compression="gzip"
                )

    #cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', '-s', default='src/sample001.mp4')
    parser.add_argument('--database', '-db' ,default='database/default.h5')

    args = parser.parse_args()
    video2hdf5(args.source_path, args.database)
