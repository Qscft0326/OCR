import os
import shutil
import re
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import cv2
import logging
logging.disable(logging.DEBUG)  
logging.disable(logging.WARNING)  
import matplotlib.pyplot as plt
def find_ocr(ocr_model,image_numpy):
    result = ocr_model.ocr(image_numpy, cls=True)
    result_ocr = "0"
    if result[0] is not None:
        result_ocr = ""
        for i_index in range(len(result[0])):
            result_ocr += result[0][i_index][1][0]
            if result_ocr.endswith('.'):
                result_ocr = result_ocr[:-1]
    result_ocr = result_ocr.replace('T', '7')
    result_ocr = re.sub(r'[^0-9.]', '', result_ocr)
    return result_ocr

def find_values(lst):
    value_dict = {}
    result = []
    for value in lst:
        if value != "0":
            if value in value_dict:
                value_dict[value] += 1
                if value_dict[value] == 3 and value not in result:
                    result.append(value)
            else:
                value_dict[value] = 1
    return result


def extract_images(video_path):
    result_ocr_list = []
    result_frame = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result_ocr= find_ocr(ocr, frame)
            result_ocr_list.append(result_ocr)
            result_frame.append(frame)
        else:
            break
    cap.release()
    return result_ocr_list,result_frame

def save_image_with_title(image, ocr_title, save_path):
    plt.imshow(image[:,:,(2,1,0)])
    plt.title(ocr_title)
    plt.savefig(save_path)
    plt.close()

ocr = PaddleOCR(use_angle_cls=False, use_gpu=True, det_model_dir="model/det", rec_model_dir="model/rec",lang="en")

img_path = "12/"
save_path = "results/"
os.makedirs(save_path,exist_ok=True)

result_ocr_list,result_frame = extract_images("12.mp4")
print("step1: get all result_ocr_list")
result_three = find_values(result_ocr_list)
print("step2: find_values")
for t in range(len(result_frame)):
    if result_ocr_list[t] in result_three:
        save_image_with_title(result_frame[t],result_ocr_list[t],save_path+str(t))
print("step3: over!")