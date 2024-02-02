import cv2
import numpy as np
import glob
import os

dataset_name = 'final_cem_obj_3'
label = '2'
path = "/home/hakertz-test/repos/6Dpose-Yolov7-Seg-AAE/yolov7/datasets/final/"
mask_filenames = glob.glob('/home/hakertz-test/repos/6Dpose-Yolov7-Seg-AAE/yolov7/datasets/final/cem_obj_3/mask/*.png')
rgb_filenames = glob.glob('/home/hakertz-test/repos/6Dpose-Yolov7-Seg-AAE/yolov7/datasets/final/cem_obj_3/rgb/*.png')
dir_path = os.path.join(path + dataset_name)
n_train_perc = 70
n_val_perc = 15
n_train = int(len(rgb_filenames) * n_train_perc / 100)
print('n_train', n_train)
n_val = int(len(rgb_filenames) * n_val_perc / 100)
print('n_val', n_val)
n_test = len(rgb_filenames) - n_train - n_val
print('n_test', n_test)

try:
    os.mkdir(dir_path)
except:
    print('This directory has already been created. New data are added to the older ones.')

train_dir = os.path.join(dir_path, 'train')
try:
    os.mkdir(train_dir)
except:
    print('This directory has already been created. New data are added to the older ones.')

val_dir = os.path.join(dir_path, 'val')
try:
    os.mkdir(val_dir)
except:
    print('This directory has already been created. New data are added to the older ones.')

test_dir = os.path.join(dir_path, 'test')
try:
    os.mkdir(test_dir)
except:
    print('This directory has already been created. New data are added to the older ones.')

print('Saving txt files with annotated masks ...')
for k, filename in enumerate(mask_filenames):
    img = cv2.imread(filename)
    img_name = label + filename[-8:-4]
    # print('img_name', img_name)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width, height = img_grey.shape
    contours, hierarchy = cv2.findContours(img_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    annotations = {}
    for i, count in enumerate(contours):
        annotations[i] = []
        for c in count:
            annotations[i].append(c[0][0] / height)
            annotations[i].append(c[0][1] / width)
    if k < n_train:
        with open(os.path.join(train_dir, img_name) + '.txt', 'w') as f:
            # print('txt filename', os.path.join(dir_path, img_name)+'.txt')
            for key in annotations.keys():
                # print('key', key)
                # print('len of annotations', len(annotations[key]))
                annotations_string = str(annotations[key]).replace(', ', ' ').replace('[', '').replace(']', '')
                f.write(label + ' ' + annotations_string + '\n')
    elif k > n_train - 1 and k < n_train + n_val:
        with open(os.path.join(val_dir, img_name) + '.txt', 'w') as f:
            # print('txt filename', os.path.join(dir_path, img_name)+'.txt')
            for key in annotations.keys():
                # print('key', key)
                # print('len of annotations', len(annotations[key]))
                annotations_string = str(annotations[key]).replace(', ', ' ').replace('[', '').replace(']', '')
                f.write(label + ' ' + annotations_string + '\n')
    else:
        with open(os.path.join(test_dir, img_name) + '.txt', 'w') as f:
            # print('txt filename', os.path.join(dir_path, img_name)+'.txt')
            for key in annotations.keys():
                # print('key', key)
                # print('len of annotations', len(annotations[key]))
                annotations_string = str(annotations[key]).replace(', ', ' ').replace('[', '').replace(']', '')
                f.write(label + ' ' + annotations_string + '\n')

print('Saving files to the right directory ...')
new_train_filenames = []
new_val_filenames = []
new_test_filenames = []
for j, filename in enumerate(rgb_filenames):
    img = cv2.imread(filename)
    img_name = label + filename[-8:-4]
    if j < n_train:
        print('rgb filename train', os.path.join(train_dir, img_name) + '.png')
        cv2.imwrite(os.path.join(train_dir, img_name) + '.png', img)
        new_train_filenames.append(os.path.join(train_dir, img_name) + '.png')
    elif j > n_train - 1 and j < n_train + n_val:
        # print('rgb filename test', os.path.join(test_dir, img_name)+'.png')
        cv2.imwrite(os.path.join(val_dir, img_name) + '.png', img)
        new_val_filenames.append(os.path.join(val_dir, img_name) + '.png')

    else:
        # print('rgb filename test', os.path.join(test_dir, img_name)+'.png')
        cv2.imwrite(os.path.join(test_dir, img_name) + '.png', img)
        new_test_filenames.append(os.path.join(test_dir, img_name) + '.png')

try:
    with open(dir_path + '/' + 'train.txt', 'a') as f:
        for line in new_train_filenames:
            f.write(line + '\n')
except:
    with open(dir_path + '/' + 'train.txt', 'w') as f:
        for line in new_train_filenames:
            f.write(line + '\n')
try:
    with open(dir_path + '/' + 'val.txt', 'a') as f:
        for line in new_val_filenames:
            f.write(line + '\n')
except:
    with open(dir_path + '/' + 'val.txt', 'w') as f:
        for line in new_val_filenames:
            f.write(line + '\n')
try:
    with open(dir_path + '/' + 'test.txt', 'a') as f:
        for line in new_test_filenames:
            f.write(line + '\n')
except:
    with open(dir_path + '/' + 'test.txt', 'w') as f:
        for line in new_test_filenames:
            f.write(line + '\n')
