#!/usr/bin/env python
import os
import os.path as osp
from glob import glob
import cv2
from functools import partial
import hashlib
import multiprocessing as mp
from tqdm import tqdm

types = {
    '2本重叠': 'spines_2',
    '2本叠加': 'spines_2',
    '封2本重叠': 'spines_2',

    '3本重叠': 'spines_3',
    '二本重叠': 'spines_2',
    '两本重叠': 'spines_2',
    '三本重叠': 'spines_3',
    '3本叠加': 'spines_3',
    '3种重叠': 'spines_3',
    '书脊': 'spine',
    '书脊侧': 'spine',
    '书籍侧': 'spine',
    '书脊页': 'spine',
    '书面侧': 'page',
    '书侧页': 'page',
    '封页侧': 'page',
    '书页侧': 'page',
    '书页册': 'page',
    '书内侧': 'page',
    '封底': 'back',
    '封面': 'front',

    '1607413651005': 'front',
    '1607413650893': 'front',
    '1607413650836': 'front',
    '1607413650610': 'front',
    '1607413650321': 'front',
    '1607413651127': 'front',
    '1607413650724': 'front',
    '1607413651299': 'front',
    '06A5915A87A1F9C7C3B874A1DC4CF2D9': 'front',
    'EB70AB4797C77037D7F3899F3806CDE0': 'front',
    '1607413651242': 'front',
    '1607413650234': 'spine',
    '1607413651184': 'front',
    '1607413651061': 'front',
    'IMG_20201209_091336': 'spines_3',
    'IMG_20201208_112344': 'spines_3',
    '1607413650949': 'front',
    '1607413650437': 'front',

    'IMG_0454': 'spines_2',
    '2本重叠 (2)': 'spines_2',
    '书页侧 (2)': 'spine',
    '封面 (2)': 'front',
}

def read_list(f):
    return set(osp.splitext(l.strip())[0].split('_')[0] for l in open(f) if l.strip())

def resize_worker(tsk, max_w):
    img_f, dst_f, tp = tsk

    img = cv2.imread(img_f)
    h, w = img.shape[:2]
    scale = max_w / max(h, w)
    if scale < 1:
        new_w, new_h = int(w*scale), int(h*scale)
        img = cv2.resize(img, (new_w, new_h))

    if 'cover' in tp:
        if w < h:
            pass

    cv2.imwrite(dst_f, img)

def hash_name(s):
    return hashlib.md5(s.encode()).hexdigest()

def parse_fname(img_f):
    book_name = osp.basename(osp.dirname(img_f))
    tp = osp.splitext(osp.basename(img_f))[0]
    if tp.endswith('_bak'):
        return

    assert tp.strip() in types, f'{img_f}'
    tp = types[tp.strip()]
    return book_name, tp

def main(src_d='images0', dst_d='images_{}x', list_f='list.txt', max_w=1024, jobs=32):
    dst_d = dst_d.format(max_w)
    os.makedirs(dst_d, exist_ok=True)

    front_bad_list = read_list('fix_lists/front/fault.txt')
    back_bad_list = read_list('fix_lists/back/angle_error.txt')
    spine_rl_list = read_list('fix_lists/spine/spine_rl.txt')

    books = {}
    for img_d in os.listdir(src_d):
        book_name = osp.basename(img_d)
        h = hash_name(book_name)
        assert h not in books
        books[h] = {
            'name': book_name,
            'hash': h,
        }

    with open(list_f, 'w') as fp:
        for k, v in books.items():
            print (k, v['name'], file=fp)

    img_fs = sorted(glob(src_d+'/*/*.jpg'))
    img_fs = [i for i in img_fs if not osp.splitext(i)[0].endswith('_bak')]

    tasks = []
    for img_f in tqdm(img_fs):
        ret = parse_fname(img_f)
        book_name, tp = parse_fname(img_f)

        cur_dst_d = osp.join(dst_d, tp)
        if tp == 'spine':
            if h in spine_rl_list:
                suffix = '_rl'
            else:
                suffix = '_lr'

            cur_dst_d = cur_dst_d + suffix

        h = hash_name(book_name)
        if tp == 'front' and h in front_bad_list:
            cur_dst_d = cur_dst_d + '_bad'
        elif tp == 'back' and h in back_bad_list:
            cur_dst_d = cur_dst_d + '_bad'

        os.makedirs(cur_dst_d, exist_ok=True)

        dst_f = osp.join(cur_dst_d, f'{h}_{tp}.jpg')
        if osp.exists(dst_f):
            continue

        tasks.append((img_f, dst_f, tp))

    with mp.Pool(jobs) as pool:
        with tqdm(total=len(tasks)) as pbar:
            for i, _ in enumerate(pool.imap_unordered(partial(resize_worker, max_w=max_w), tasks)):
                pbar.update()

if __name__ == "__main__":
    import fire
    fire.Fire(main)
