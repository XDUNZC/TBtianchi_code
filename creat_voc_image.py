import json
import csv
import cv2
from PIL import Image
import PIL
import random
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
labels = ('长马甲', '古风', '短马甲', '背心上衣', '背带裤', '连体衣', '吊带上衣', '中裤', '短袖衬衫', '无袖上衣',
                 '长袖衬衫', '中等半身裙', '长半身裙', '长外套', '短裙', '无袖连衣裙', '短裤', '短外套',
                 '长袖连衣裙', '长袖上衣', '长裤', '短袖连衣裙', '短袖上衣')


def creat_voc_image_xml(file_list_path='data/train_down_sample.csv',save_path='data/JPEGImages/'):
    '''
    In the first time
    Do not forget  mkdir JPEGImages and mkdir labels in data/
    --------------------------------------------------------

    --------------------------------------------------------


    :param file_list_path:
    :param save_path:
    :return:
    '''
    if not os.path.exists(file_list_path):
        print('There is not the file: ',file_list_path)
    with open(file_list_path) as f:
        reader = csv.reader(f)
        next(reader)
        # num = 0
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
                with open(ann, 'r') as f:
                    ann = json.load(f)['annotations']
            box_with_label=[]
            for item in ann:
                box_with_label.append(item['box']+','+ann['label'])
            width,height =  image.size
            save_xml(save_path+name,box_with_label,width=width,height=height,save_dir='data/Annotations')
            if not os.path.exists(save_path+name):
                image.save(save_path+name,quality=95,subsampling=0)


    print('create finished!')



def creat_voc_txt(file_list_path='data/train_down_sample.csv',save_path='data/JPEGImages/',save_txt_path='data/labels/trainval.txt'):
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
        # num = 0
        for row in reader:
            # data/validation_dataset_part1/image/064598/0.jpg,data/validation_dataset_part1/image_annotation/064598/0.json,-1
            # data/validation_dataset_part1/video/076751.mp4,data/validation_dataset_part1/video_annotation/076751.json,40
            image, ann, frame = row
            frame = int(frame)
            if frame != -1:
                name = image.split('/')[-1].split('.')[0]+'_'+('%02d' %frame)+'_v'
            else:
                name = image.split('/')[-2]+'_'+('%02d' %int(image.split('/')[-1].split('.')[0]))+'_i'
            ann_in_txt = '\n'

            ftrainval.write(name+ann_in_txt)
    ftrainval.close()
    print('create finished!')



def split_train_and_val_dataset(file_path='data/labels/trainval.txt',train_path='data/labels/train.txt',val_path='data/labels/val.txt'):
    f = open(file_path,'r')
    train_f = open(train_path,'w')
    val_f = open(val_path,'w')
    data_list = f.readlines()
    random.shuffle(data_list)
    N = len(data_list)
    train_txt = data_list[:int(N*0.8)]
    val_txt = data_list[int(N*0.8):]
    train_f.writelines(train_txt)
    val_f.writelines(val_txt)
    f.close()
    train_f.close()
    val_f.close()
    print('Have created train and val txt! ')

def split_fake_train_val_test_dataset(file_path='data/labels/trainval.txt',train_path='data/labels/fake_train.txt',val_path='data/labels/fake_val.txt',
                                      test_path='data/labels/fake_test.txt' ):
    f = open(file_path,'r')
    train_f = open(train_path,'w')
    val_f = open(val_path,'w')
    test_f = open(test_path,'w')
    data_list = f.readlines()
    random.shuffle(data_list)
    N = len(data_list)
    train_txt = data_list[:int(N*0.0001)]
    val_txt = data_list[int(N*0.0001):int(N*0.0001)+70]
    test_txt = data_list[int(N*0.0001)+100:int(N*0.0001)+170]
    train_f.writelines(train_txt)
    val_f.writelines(val_txt)
    test_f.writelines(test_txt)
    f.close()
    train_f.close()
    val_f.close()
    test_f.close()
    print('Have created train and val txt! ')








def save_xml(image_path, bbox_label, width,height,save_dir='data/Annotations', channel=3):
    '''
      :param image_path:图片名 'data/JPEGImages/034227_00_i.jpg'
      :param bbox:对应的bbox ['x,y,x,y,label','...'...]
      :param save_dir: 存xml文件的地址
      :param channel:这个是图片的通道
      :return:
    '''
    image_name = image_path.split('/')[-1]

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for i in bbox_label:
        left, top, right, bottom, label = i.split(',')
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True,encoding = "utf-8")
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return



def creat_voc_ann(file_path='data/labels/trainval.txt'):
    f = open(file_path)
    file_list = f.readlines()

    # flag = 0

    for file in file_list:
        image_path = file.split(' ')[0]
        bbox_label = file.split(' ')[1:]
        save_xml(image_path,bbox_label)

        # flag+=1
        # if flag>20:
        #     break

    f.close()
    return

def main():
    '''
    巡行前生成data/Annotations 和data/labels 文件夹

    :return:
    '''
    # 生成voc的图片和xml
    creat_voc_image_xml()
    creat_voc_image_xml(file_list_path='data/val_down_sample.csv')

    # 生成voc的txt
    creat_voc_txt()
    creat_voc_txt(file_list_path='data/val_down_sample.csv')

    # 将trainval.txt 划分成为训练集和测试集
    split_train_and_val_dataset()

    # split_fake_train_val_test_dataset()
if __name__ == '__main__':
    main()
