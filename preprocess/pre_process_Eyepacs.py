import torch

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

from PIL import Image

DATA_PATH_META = '/l/users/xiangjian.hou/hc701-fed_data/eyepacs/diabetic-retinopathy-detection/'
SAVE_PATH_META = '/l/users/xiangjian.hou/preprocessed/eyepacs/'

class pre_process_messidor(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_path = os.path.join(self.data_path, 'train')
        self.test_path = os.path.join(self.data_path, 'test')
        self.train_info = pd.read_csv(os.path.join(self.data_path, 'trainLabels.csv'))

    def process_train(self, pre_processed_path):
        save_path = os.path.join(pre_processed_path, 'train')
        for i,j in zip(self.train_info['image'],self.train_info['level']):
            pre_processed_data = {}
            origin_data = Image.open(os.path.join(self.train_path, i+'.jpeg'))
            gray = cv2.cvtColor(np.array(origin_data), cv2.COLOR_BGR2GRAY)
            # Threshold the image to remove the black edges
            _, thresholded = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
            print(f'image: {i}, image_max: {np.max(gray)}, image_min: {np.min(gray)}')
            # Find the contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the bounding box of the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Crop the image to the ROI
            cropped = np.array(origin_data)[y:y+h, x:x+w]
            # show the image
            resized = cv2.resize(cropped, (224, 224))
            pre_processed_data['image'] = resized
            pre_processed_data['label'] = j
            np.save(os.path.join(save_path, i), pre_processed_data)

    def process_test(self, pre_processed_path):
        save_path = os.path.join(pre_processed_path, 'test')
        for i in tqdm(os.listdir(self.test_path)):
            pre_processed_data = {}
            origin_data = Image.open(os.path.join(self.test_path, i))
            gray = cv2.cvtColor(np.array(origin_data), cv2.COLOR_BGR2GRAY)
            # Threshold the image to remove the black edges
            _, thresholded = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
            # Find the contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the bounding box of the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Crop the image to the ROI
            cropped = np.array(origin_data)[y:y+h, x:x+w]
            # show the image
            resized = cv2.resize(cropped, (224, 224))
            pre_processed_data['image'] = resized
            np.save(os.path.join(save_path, i.split('.')[0]), pre_processed_data)


if __name__ == '__main__':
    pre_process = pre_process_messidor(DATA_PATH_META)
    pre_process.process_train(SAVE_PATH_META)
    pre_process.process_test(SAVE_PATH_META)