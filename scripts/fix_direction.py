#!/usr/bin/env python
import os.path as osp
from glob import glob
from tqdm import tqdm
import numpy as np
import shutil
import cv2
import pylab as plt
import multiprocessing as mp
from preproc_images import parse_fname, hash_name, read_list

def rot90(v, n):
    for i in range(n):
        v = np.rot90(v)
    return v

def proc_worker(tsk):
    src_f, dst_f, mode = tsk
    img = cv2.imread(src_f)
    if mode == 'down':
        n = 2
    elif mode == 'right':
        n = 1
    elif mode == 'left':
        n = 3
    else:
        raise

    img = rot90(img, n)
    cv2.imwrite(dst_f, img)

def main(jobs=32):
    src_d = 'images0'
    img_fs = sorted(glob(src_d+'/*/*.jpg'))

    front_down_list = read_list('fix_lists/front/down.txt')
    front_right_list = read_list('fix_lists/front/right.txt')
    front_left_list = read_list('fix_lists/front/left.txt')

    back_down_list = read_list('fix_lists/back/down.txt')
    back_left_list = read_list('fix_lists/back/left.txt')
    back_right_list = read_list('fix_lists/back/right.txt')

    all_list = front_down_list | front_right_list | front_left_list | back_down_list | back_left_list | back_right_list

    tasks = []
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

        if tp in ('front', 'back'):
            if tp == 'front':
                down_l, right_l, left_l = front_down_list, front_right_list, front_left_list,
            elif tp == 'back':
                down_l, right_l, left_l = back_down_list, back_right_list, back_left_list,
            mode = ''
            if h in down_l:
                mode = 'down'
            elif h in right_l:
                mode = 'right'
            elif h in left_l:
                mode = 'left'
            else:
                continue

            tasks.append((bak_f, img_f, mode))

    with mp.Pool(jobs) as pool:
        with tqdm(total=len(tasks)) as pbar:
            for i, _ in enumerate(pool.imap_unordered(proc_worker, tasks)):
                pbar.update()

if __name__ == "__main__":
    import fire
    fire.Fire(main)
