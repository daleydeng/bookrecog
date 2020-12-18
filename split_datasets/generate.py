import os
import os.path as osp
import sys
sys.path.append(os.getcwd())

import json
import click
import shutil
import multiprocessing as mp

from tqdm import tqdm
from functools import partial
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    data=[]
    imgs_nm = os.listdir(data_dir)
    for img_nm in imgs_nm:
        img_p = osp.join(data_dir, img_nm)
        data.append(img_p)
    return data


def split_train_eval_test(data, scale = [7, 2, 1], random_state=42):
    '''
    we split the dataset by a ratio of 7:2:1
    the ratio of 6：2：2 is also commonly used to split the dataset
    '''
    train_eval_data, test_data = train_test_split(data, test_size=scale[2]/(scale[0]+scale[1]+scale[2]), random_state=42)
    train_data, eval_data = train_test_split(train_eval_data, test_size=scale[1]/(scale[0]+scale[1]), random_state=42)
    print(f"train : eval : test = {len(train_data)} : {len(eval_data)} : {len(test_data)}")

    return train_data, eval_data, test_data


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def write_txt(data, txt_file, class_indict):
    with open(txt_file,'w') as f:
        for img_p in data:
            root, img_nm = osp.split(img_p)
            cla = osp.split(root)[-1]
            cla_nb = get_key(class_indict, cla)[0]
            f.writelines(img_p + '\x20' + cla_nb) 
            f.write('\n')
    print(f"write '{txt_file}' done!")


def copy_image(img_p, dst_dir):
    root, img_nm = osp.split(img_p)
    cla = osp.split(root)[-1]
    dst = osp.join(dst_dir, cla)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(img_p, dst)
    return osp.join(dst,img_nm)


@click.command()
@click.option('--class_json_path', default='./classes/class_indices.json', help='out test data')
@click.option('--data_dirs', default=['./data/page','./data/spine'], help='original data')
@click.option('--train_dir', default='./datasets/train', help='out train data')
@click.option('--eval_dir', default='./datasets/eval', help='out eval data')
@click.option('--test_dir', default='./datasets/test', help='out test data')
@click.option('--save_txt', default=True, help='save tran/eval/test.txt or not')
@click.option('--jobs', default=mp.cpu_count())
def main(class_json_path, data_dirs, train_dir, eval_dir, test_dir, save_txt, jobs):
    # load original data
    data=[]
    for data_dir in data_dirs:
        data += load_data(data_dir)

    # classes = {'0':'page', '1':'spine'}
    json_file = open(class_json_path, 'rb')
    class_indict = json.load(json_file)

    # split
    train_data, eval_data, test_data = split_train_eval_test(data, scale = [7, 2, 1], random_state=42)

    train_worker = partial(copy_image, dst_dir = train_dir)
    new_train_data = []
    with mp.Pool(jobs) as p:
        with tqdm(total=len(train_data)) as pbar:
            for i, r in enumerate(p.imap_unordered(train_worker, train_data)):
                new_train_data.append(r)
                pbar.update()

    eval_worker = partial(copy_image, dst_dir = eval_dir)
    new_eval_data = []
    with mp.Pool(jobs) as p:
        with tqdm(total=len(eval_data)) as pbar:
            for i, r in enumerate(p.imap_unordered(eval_worker, eval_data)):
                new_eval_data.append(r)
                pbar.update()

    test_worker = partial(copy_image, dst_dir = test_dir)
    new_test_data = []
    with mp.Pool(jobs) as p:
        with tqdm(total=len(test_data)) as pbar:
            for i, r in enumerate(p.imap_unordered(test_worker, test_data)):
                new_test_data.append(r)
                pbar.update()

    print(save_txt)
    if save_txt:
        train_txt = osp.join(train_dir,'train.txt')
        eval_txt = osp.join(eval_dir,'eval.txt')
        test_txt = osp.join(test_dir,'test.txt')

        write_txt(new_train_data, train_txt, class_indict)
        write_txt(new_eval_data, eval_txt, class_indict)
        write_txt(new_test_data, test_txt, class_indict)
    print("done.")


if __name__ == "__main__":
    main()


