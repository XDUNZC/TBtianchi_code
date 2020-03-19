import json
import glob
import pandas as pd
import numpy as np
import random
import colorsys
import csv
import os
import cv2
from PIL import Image
import PIL

label2num = {'长马甲': 0, '古装': 1, '短马甲': 2, '背心上衣': 3, '背带裤': 4, '连体衣': 5, '吊带上衣': 6, '中裤': 7, '短袖衬衫': 8, '无袖上衣': 9,
                 '长袖衬衫': 10, '中等半身裙': 11, '长半身裙': 12, '长外套': 13, '短裙': 14, '无袖连衣裙': 15, '短裤': 16, '短外套': 17,
                 '长袖连衣裙': 18, '长袖上衣': 19, '长裤': 20, '短袖连衣裙': 21, '短袖上衣': 22, '古风': 1}


def creat_voc_image(file_list_path='data/train_down_sample.csv',save_path='data/JPEGImages/',save_txt_path='data/labels/trainval.txt'):
    '''
    In the first time
    Do not forget  mkdir JPEGImages and mkdir labels in data/
    --------------------------------------------------------
    if you want to creat test.txt
    make:
    file_list_path='data/val_down_sample.csv'
    save_txt_path='data/labels/test.txt'
    --------------------------------------------------------


    :param file_list_path:
    :param save_path:
    :param save_txt_path:
    :return:
    '''
    if not os.path.exists(file_list_path):
        print('There is not the file: ',file_list_path)
    ftrainval = open(save_txt_path, 'w')
    with open(file_list_path) as f:
        reader = csv.reader(f)
        next(reader)
        num = 0
        for row in reader:
            # data/validation_dataset_part1/image/064598/0.jpg,data/validation_dataset_part1/image_annotation/064598/0.json,-1
            # data/validation_dataset_part1/video/076751.mp4,data/validation_dataset_part1/video_annotation/076751.json,40
            image, ann, frame = row
            frame = int(frame)
            if frame != -1:
                name = image.split('/')[-1].split('.')[0]+'_'+('%02d' %frame)+'_v.jpg'
                cap = cv2.VideoCapture(image)
                # print(frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                flag, image = cap.read()
                assert flag == 1
                image = image[:, :, ::-1].copy()
                image = PIL.Image.fromarray(image)
                with open(ann, 'r') as f:
                    video_anns = json.load(f)
                ann = video_anns['frames'][frame // 40]['annotations']
            else:
                name = image.split('/')[-2]+'_'+('%02d' %int(image.split('/')[-1].split('.')[0]))+'_i.jpg'
                image = Image.open(image).convert('RGB')
                # image = cv2.imread(image)
                # image = image[:, :, ::-1]
                with open(ann, 'r') as f:
                    ann = json.load(f)['annotations']
            ann_in_txt = ''
            for i in ann:
                # print(",".join([str(j) for j in i['box']]))
                ann_in_txt+= ' '+ ",".join([str(j) for j in i['box']]) + ','+str(label2num[i['label']])
            ann_in_txt+='\n'
            image.save(save_path+name,quality=95,subsampling=0)
            ftrainval.write(save_path+name+ann_in_txt)
            num+=1
            # if num>20:
            #     break
    ftrainval.close()
    print('create finished!')

def main():
    creat_voc_image()

if __name__ == '__main__':
    main()
