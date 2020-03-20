##########
translate voc to yolov3 code dataset
##########
import cv2


def translate_txt(trainval_path='./data/labels/trainval.txt',train_file_path='./data/',labels_path='./data/labels/',model='train')
    # limit
    num=0
    # index file
    index_file = open(train_file_path+model+'.txt',"w")
    with open(trainval_path, "r") as f:  # 打开文件
        trainval = f.readlines()  # 读取文件
        for line in trainval:
            if line=='':
                break
            line = line.strip('\n')
            line=line.split(' ')
            img_path = line[0]
            # write index file
            index_file.write(img_path+'\n')
            # write label file
            img_name = img_path.splite('/')[-1].split('.')[0]
            with open(labels_path+img_name+'.txt','w') as label_file:
                # read img and get w,h
                img = cv2.imread(dirfile)
                img_h,img_w = img.shape[:-1]
                # split box
                line=line[1:]
                for box in line:
                    if box =='':
                        break
                    box=box.split(',')
                    new_box=''
                    # add label
                    new_box=new_box+box[4]
                    # scale to [0,1]
                    x1=float(box[0])
                    y1=float(box[1])
                    x2=float(box[2])
                    y2=float(box[3])
                    x=(x1+x2)/2/img_w
                    y=(y1+y2)/2/img_h
                    w=(x2-x1)/img_w
                    h=(y2-y1)/img_h
                    # write
                    new_box=new_box+str(x)+' '+str(y)+' '
                    new_box=new_box+str(w)+' '+str(h)+'\n'
        num+=1
        if num==20:
            break

    index_file.close()








def main():
    translate_txt()


if __name__ == '__main__':
    main()
