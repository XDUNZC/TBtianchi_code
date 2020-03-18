import json
import glob
import pandas as pd
import numpy as np
import random
import colorsys
def get_fixed_list():
    aug_label = ['古风', '长马甲', '古装', '短马甲', '背心上衣', '背带裤', '连体衣', '吊带上衣', '中裤', '短袖衬衫', '无袖上衣', '长袖衬衫']
    label2num = {'长马甲': 0, '古装': 1, '短马甲': 2, '背心上衣': 3, '背带裤': 4, '连体衣': 5, '吊带上衣': 6, '中裤': 7, '短袖衬衫': 8, '无袖上衣': 9,
                 '长袖衬衫': 10, '中等半身裙': 11, '长半身裙': 12, '长外套': 13, '短裙': 14, '无袖连衣裙': 15, '短裤': 16, '短外套': 17,
                 '长袖连衣裙': 18, '长袖上衣': 19, '长裤': 20, '短袖连衣裙': 21, '短袖上衣': 22, '古风': 1}
    data_root_path = 'data/'
    return aug_label,label2num,data_root_path

def creat_csv_downsample(aug_label,data_root_path='data/',mode='train'):
    dataset_paths = glob.glob(data_root_path+mode+'*')
    # 图像库中标注
    img_ann_folder_paths = []  # 所有data/train_dataset_part<n>/image_annotatonl中所有文件夹

    # 视频库中标注
    video_ann_paths = []  # 所有data/train_dataset_part<n>/video_annotation中所有json文件


    for dataset_path in dataset_paths:
        img_ann_folder_paths.extend(glob.glob(dataset_path + '/image_annotation/*'))

        video_ann_paths.extend(glob.glob(dataset_path + '/video_annotation/*.json'))

    image_db = []
    for img_ann_folder_path in img_ann_folder_paths[:]:
        split_list = img_ann_folder_path.split('/')
        img_folder_path = 'data/' + split_list[1] + '/image/' + split_list[-1] + '/'
        json_paths = glob.glob(img_ann_folder_path + '/*.json')
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                img_anns = json.load(f)
            if len(img_anns['annotations']) > 0:
                flag = 0
                for img_ann in img_anns['annotations']:
                    if img_ann['label'] not in aug_label:
                        flag = 1
                        break
                img_path = img_folder_path + json_path.split('/')[-1].split('.')[0] + '.jpg'
                image_db.append([img_path, json_path, -1])
                if flag:
                    break
    image_db = pd.DataFrame(image_db, columns=['file', 'ann', 'frame'])

    video_db = []

    for json_path in video_ann_paths[:]:
        with open(json_path, 'r') as f:  # 'data/train_dataset_part3/video_annotation/002061.json'
            v_ann = json.load(f)
        split_list = json_path.split('/')
        img_folder_path = 'data/' + split_list[1] + '/video/' + split_list[-1].split('.')[0] + '.mp4'
        for fram in v_ann['frames']:
            if len(fram['annotations']) > 0:
                flag = 0
                for img_ann in fram['annotations']:
                    if img_ann['label'] not in aug_label:
                        flag = 1
                        break
                frame_index = fram['frame_index']
                video_db.append([img_folder_path, json_path, frame_index])
                if flag:
                    break

    video_db = pd.DataFrame(video_db, columns=['file', 'ann', 'frame'])
    train_db = pd.concat([image_db, video_db])
    assert len(train_db) == len(image_db) + len(video_db)
    train_db.to_csv(data_root_path+mode+'_down_sample.csv', index=False)
    print('已生成csv路径文件：' + data_root_path+mode+'_down_sample.csv')
    print(train_db.info())

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def draw_boxs(image, boxs, colors):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    # print(colors)
    for i,box in enumerate(boxs):
        x1, y1, x2, y2 = box
        # print(x1, y1, x2, y2 )
        # print(image[:,y1:y1 + 2,x1:x2].shape)
        # print(colors[i])
        image[:,y1:y1 + 2, x1:x2] = colors[i]
        image[:,y2:y2 + 2, x1:x2] = colors[i]
        image[:,y1:y2, x1:x1 + 2] = colors[i]
        image[:,y1:y2, x2:x2 + 2] = colors[i]
    return image

def convert_annotations_to_boxs(anns,label2num):
    boxs = []
    for ann in anns:
        print(ann)
        box = ann['box']
        # box.append(label2num[ann['label']])
        boxs.append(box)
    return boxs


def main():
    auglabel,_,_ = get_fixed_list()
    print(len(creat_csv_downsample(auglabel,mode='val')))

if __name__ == '__main__':
    main()