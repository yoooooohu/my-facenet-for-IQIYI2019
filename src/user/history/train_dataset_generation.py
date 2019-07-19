# -*- coding: utf-8 -*-
# command: python src/user/train_dataset_generation.py

import data_processing_AIQIYI as dpa
import sys
import os
import shutil
import argparse

# ground truth paths
train_gt_path = './offical_files/train_gt.txt'

log_path = './logs/user/train_dataset_generation_log.txt'

def main(args):
    train_frame_output_path = args.face_dir + '/train_frame'
    train_frame_1_path = args.face_dir + '/train_frame_1'
    train_frame_2_path = args.face_dir + '/train_frame_2'
    train_frame_3_path = args.face_dir + '/train_frame_3'   
     
    for i in range(10034):
        tmp_path = train_frame_output_path + '/{}'.format(i+1)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

    gt_dict = dpa.get_train_image_labels(train_gt_path)


    for root,dirs,_ in os.walk(train_frame_1_path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            tar_dir = os.path.join(train_frame_output_path, gt_dict[dir])

            for _,_,files in os.walk(src_dir):
                for file in files:
                    shutil.copyfile(os.path.join(src_dir,file), os.path.join(tar_dir, dir + '.' + file))
    print('{} had been moved.'.format(train_frame_1_path))

    for root,dirs,_ in os.walk(train_frame_2_path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            tar_dir = os.path.join(train_frame_output_path, gt_dict[dir])

            for _,_,files in os.walk(src_dir):
                for file in files:
                    shutil.copyfile(os.path.join(src_dir,file), os.path.join(tar_dir, dir + '.' + file))
    print('{} had been moved.'.format(train_frame_2_path))

    for root,dirs,_ in os.walk(train_frame_3_path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            tar_dir = os.path.join(train_frame_output_path, gt_dict[dir])

            for _,_,files in os.walk(src_dir):
                for file in files:
                    shutil.copyfile(os.path.join(src_dir,file), os.path.join(tar_dir, dir + '.' + file))
    print('{} had been moved.'.format(train_frame_3_path))

    for root,dirs,_ in os.walk(train_frame_output_path): 
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            for _,_,files in os.walk(src_dir):
                if len(files) == 0:
                    dpa.log_write('{} do not have any face'.format(dir), log_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--face_dir', type=str,
        help='The location of output face features set.', default='./datasets/IQIYIdataset0')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
