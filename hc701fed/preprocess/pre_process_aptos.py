import torch

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

RAW_TRAIN_DATA = '/home/hong/hc701/raw_data/aptosdata/train_images/train_images'
RAW_TRAIN_INFO = '/home/hong/hc701/raw_data/aptosdata/train_1.csv'

RAW_TEST_DATA = '/home/hong/hc701/raw_data/aptosdata/test_images/test_images'
RAW_TEST_INFO = '/home/hong/hc701/raw_data/aptosdata/test.csv'

RAW_VAL_DATA = '/home/hong/hc701/raw_data/aptosdata/val_images/val_images'
RAW_VAL_INFO = '/home/hong/hc701/raw_data/aptosdata/valid.csv'

SAVE_TRAIN_DATA = '/home/hong/hc701/preprocessed/aptos/train'
SAVE_TEST_DATA = '/home/hong/hc701/preprocessed/aptos/test'
SAVE_VAL_DATA = '/home/hong/hc701/preprocessed/aptos/val'

class preprocess_aptos(object):
    def __init__(self, raw_train_data, raw_train_info, raw_test_data, raw_test_info, raw_val_data, raw_val_info,
                 save_train_data, save_test_data, save_val_data):
        self.raw_train_data = raw_train_data
        self.raw_train_info = raw_train_info
        self.raw_test_data = raw_test_data
        self.raw_test_info = raw_test_info
        self.raw_val_data = raw_val_data
        self.raw_val_info = raw_val_info
        self.save_train_data = save_train_data
        self.save_test_data = save_test_data
        self.save_val_data = save_val_data

    def preprocess_train(self):
        train_info = pd.read_csv(self.raw_train_info)
        train_info = train_info.dropna()
        train_info['diagnosis'] = train_info['diagnosis'].astype(int)
        train_info['id_code'] = train_info['id_code'].astype(str)
        for i in tqdm(range(len(train_info['id_code']))):
            pre_processed_data = {}
            image = cv2.imread(os.path.join(self.raw_train_data, train_info['id_code'][i]+'.png'))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            croppped = image[y:y + h, x:x + w]
            resized = cv2.resize(croppped, (224, 224))
            pre_processed_data['image']=resized
            pre_processed_data['label'] = train_info['diagnosis'][i]
            np.save(os.path.join(self.save_train_data, train_info['id_code'][i]), pre_processed_data)




    def preprocess_test(self):
        test_info = pd.read_csv(self.raw_test_info)
        test_info = test_info.dropna()
        test_info['diagnosis'] = test_info['diagnosis'].astype(int)
        test_info['id_code'] = test_info['id_code'].astype(str)
        for i in tqdm(range(len(test_info['id_code']))):
            pre_processed_data = {}
            image = cv2.imread(os.path.join(self.raw_test_data, test_info['id_code'][i]+'.png'))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            croppped = image[y:y + h, x:x + w]
            resized = cv2.resize(croppped, (224, 224))
            pre_processed_data['image']=resized
            pre_processed_data['label'] = test_info['diagnosis'][i]
            np.save(os.path.join(self.save_test_data, test_info['id_code'][i]), pre_processed_data)

    def preprocess_val(self):
        val_info = pd.read_csv(self.raw_val_info)
        val_info = val_info.dropna()
        val_info['diagnosis'] = val_info['diagnosis'].astype(int)
        val_info['id_code'] = val_info['id_code'].astype(str)
        for i in tqdm(range(len(val_info['id_code']))):
            pre_processed_data = {}
            image = cv2.imread(os.path.join(self.raw_val_data, val_info['id_code'][i]+'.png'))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            croppped = image[y:y + h, x:x + w]
            resized = cv2.resize(croppped, (224, 224))
            pre_processed_data['image']=resized
            pre_processed_data['label']=val_info['diagnosis'][i]
            np.save(os.path.join(self.save_val_data, val_info['id_code'][i]), pre_processed_data)


if __name__ == '__main__':
    pre = preprocess_aptos(RAW_TRAIN_DATA, RAW_TRAIN_INFO, RAW_TEST_DATA, RAW_TEST_INFO, RAW_VAL_DATA, RAW_VAL_INFO,
                           SAVE_TRAIN_DATA, SAVE_TEST_DATA, SAVE_VAL_DATA)
    pre.preprocess_train()
    pre.preprocess_test()
    pre.preprocess_val()
    