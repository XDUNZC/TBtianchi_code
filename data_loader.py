import torch
import os
import random, csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms
from PIL import Image
import PIL
import torchvision.transforms.functional as TF
import cv2

from utils import creat_csv_downsample,get_fixed_list,random_colors,draw_boxs,convert_annotations_to_boxs
def my_collate(batch):
    data= [x[0] for x in batch]
    data = torch.stack(data, 0, out=None)  # just form a list of tensor

    target = [item[1] for item in batch]
    return [data, target]

class My_DataLoader(Dataset):

    def __init__(self, root, resize, mode):
        super(My_DataLoader, self).__init__()

        self.root = root  # 'data/'
        self.mode = mode # 'train' / 'val
        self.resize = resize
        self.aug_label,self.label2num,_ = get_fixed_list()
        # images_t0, images_t1, images_ground_truth
        self.images, self.anns, self.frames = self.load_csv( mode+'_down_sample.csv')

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):

            creat_csv_downsample(self.aug_label,self.root,mode=self.mode)
        # read from csv file
        images, anns, frames = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                # data/validation_dataset_part1/image/064598/0.jpg,data/validation_dataset_part1/image_annotation/064598/0.json,-1
                image, ann, frame = row

                images.append(image)
                anns.append(ann)
                frames.append(int(frame))
        assert len(images) == len(anns) == len(frames)
        return images, anns, frames

    def __len__(self):

        return len(self.images)

    def augmentation(self, image, ann, frame):
        if frame!=-1:
            cap = cv2.VideoCapture(image)
            # print(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            flag,image=cap.read()
            assert flag==1
            image = image[:,:,::-1].copy()
            image = PIL.Image.fromarray(image)
            with open(ann, 'r') as f:
                video_anns = json.load(f)
            ann = video_anns['frames'][frame//40]['annotations']
        else:
            image = Image.open(image).convert('RGB')
            # image = cv2.imread(image)
            # image = image[:, :, ::-1]
            with open(ann, 'r') as f:
                ann = json.load(f)['annotations']

        # Random crop
        if self.mode == 'train':
            pass
        # resize
        image = TF.resize(image,self.resize)
        # Transform to tensor
        image = TF.to_tensor(image)

        # Normalize Image
        normal = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        image = normal(image)
        ann = ann

        return image, ann

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x
# TODO 这里有个bug需要修复
    def apply_box(self,image,ann):
        if image.dim()==4:
            for i in range(image.shape[0]):
                img = image[i]
                ann_sigle = ann[i]
                img = self.denormalize(img)
                N = len(ann_sigle)
                colors = torch.tensor(random_colors(N))
                colors = colors.unsqueeze(-1).unsqueeze(-1)
                print(ann_sigle)
                boxs = convert_annotations_to_boxs(ann_sigle, self.label2num)
                img_with_boxs = draw_boxs(img, boxs, colors)
                image[i] = img_with_boxs
            return image
        else:
            image = self.denormalize(image)
            N = len(ann)
            colors = torch.tensor(random_colors(N))
            colors = colors.unsqueeze(-1).unsqueeze(-1)
            boxs = convert_annotations_to_boxs(ann,self.label2num)
            image_with_boxs = draw_boxs(image,boxs,colors)
            return image_with_boxs

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        image, ann, frame = self.images[idx], self.anns[idx], self.frames[idx]

        image, ann = self.augmentation(image, ann, frame)

        assert type(ann) == list
        return image, ann


def main():
    # import visdom
    import time
    import torchvision

    # viz = visdom.Visdom()

    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)
    #
    # for x,y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)
    mode = ['train','val','test'][0]
    db = My_DataLoader('data/', (800, 800), mode)

    x, y = next(iter(db))
    print('sample:', x.shape,np.shape(y))
    print(y)
    #
    # viz.image(db.apply_box(x,y), win='sample_image', opts=dict(title='sample_image'))
    # viz.text(str(y), win='sample_annotations', opts=dict(title='sample_annotations'))
    #
    loader = DataLoader(db, batch_size=2, shuffle=True, num_workers=8,collate_fn=my_collate)
    print('mode: ', mode, 'length: ', len(loader))

    for x, y in loader:
        print('sample: ', x.shape, np.shape(y))
        print(y)

        # viz.images(db.apply_box(x,y), nrow=8, win='sample_image', opts=dict(title='sample_image'))
        # viz.text(str(y), win='sample_annotations', opts=dict(title='sample_annotations'))
        # time.sleep(10)


if __name__ == '__main__':
    main()

