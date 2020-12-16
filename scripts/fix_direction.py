#!/usr/bin/env python
import os.path as osp
from glob import glob
from tqdm import tqdm
import numpy as np
import shutil
import cv2
import pylab as plt
from preproc_images import parse_fname, hash_name, read_list

def rot90(v, n):
    for i in range(n):
        v = np.rot90(v)
    return v

def main():
    src_d = 'images0'
    img_fs = sorted(glob(src_d+'/*/*.jpg'))

    front_down_list = read_list('fix_lists/front/down.txt')
    front_right_list = read_list('fix_lists/front/right.txt')
    front_left_list = read_list('fix_lists/front/left.txt')

    back_down_list = read_list('fix_lists/back/down.txt')
    back_left_list = read_list('fix_lists/back/left.txt')
    back_right_list = read_list('fix_lists/back/right.txt')

    all_list = front_down_list | front_right_list | front_left_list | back_down_list | back_left_list | back_right_list

    for img_f in tqdm(img_fs):
        ret = parse_fname(img_f)
        if not ret:
            continue

        book_name, tp = ret
        h = hash_name(book_name)

        if not h in all_list:
            continue

        bak_f = osp.splitext(img_f)[0]+'_bak.jpg'
        if not osp.exists(bak_f):
            shutil.copy2(img_f, bak_f)

        for pre_tp, down_l, right_l, left_l in [
                [
                    'front',
                    front_down_list,
                    front_right_list,
                    front_left_list,
                ], [
                    'back',
                    back_down_list,
                    back_right_list,
                    back_left_list,
                ]]:

            if tp != pre_tp:
                continue

            if h in down_l:
                img = cv2.imread(bak_f)
                img = rot90(img, 2)
                cv2.imwrite(img_f, img)

            elif h in right_l:
                img = cv2.imread(bak_f)
                img = rot90(img, 1)
                cv2.imwrite(img_f, img)

            elif h in left_l:
                img = cv2.imread(bak_f)
                img = rot90(img, 3)
                cv2.imwrite(img_f, img)


if __name__ == "__main__":
    main()
