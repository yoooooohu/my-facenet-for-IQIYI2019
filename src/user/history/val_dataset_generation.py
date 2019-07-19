# -*- coding: utf-8 -*-
# command: python src/user/train_dataset_generation.py

import data_processing_AIQIYI as dpa
import sys
import os
import shutil
import argparse

# ground truth paths
val_gt_path = './offical_files/val_gt.txt'

val_log_path = './logs/user/val_dataset_generation_log.txt'

def main(args):
    val_frame_output_path = args.face_dir + '/val_frame'
    val_frame_1_path = args.face_dir + '/val_frame_1'
    val_frame_2_path = args.face_dir + '/val_frame_2'
    val_frame_3_path = args.face_dir + '/val_frame_3'
    val_frame_4_path = args.face_dir + '/val_frame_4'   
     
    for i in range(10034):
        tmp_path = val_frame_output_path + '/{}'.format(i+1)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

    gt_dict = dpa.get_val_image_labels(val_gt_path)


    for root,dirs,_ in os.walk(val_frame_1_path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for dir in dirs:
            if dir not in gt_dict.keys():
                continue
            src_dir = os.path.join(root, dir)
            tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])

            for _,_,files in os.walk(src_dir):
                for file in files:
                    shutil.copyfile(os.path.join(src_dir,file), os.path.join(tar_dir, dir + '.' + file))
    print('{} had been moved.'.format(val_frame_1_path))

    for root,dirs,_ in os.walk(val_frame_2_path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for dir in dirs:
            if dir not in gt_dict.keys():
                continue            
            src_dir = os.path.join(root, dir)
            tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])

            for _,_,files in os.walk(src_dir):
                for file in files:
                    shutil.copyfile(os.path.join(src_dir,file), os.path.join(tar_dir, dir + '.' + file))
    print('{} had been moved.'.format(val_frame_2_path))

    for root,dirs,_ in os.walk(val_frame_3_path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for dir in dirs:
            if dir not in gt_dict.keys():
                continue            
            src_dir = os.path.join(root, dir)
            tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])

            for _,_,files in os.walk(src_dir):
                for file in files:
                    shutil.copyfile(os.path.join(src_dir,file), os.path.join(tar_dir, dir + '.' + file))
    print('{} had been moved.'.format(val_frame_3_path))

    for root,dirs,_ in os.walk(val_frame_4_path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for dir in dirs:
            if dir not in gt_dict.keys():
                continue            
            src_dir = os.path.join(root, dir)
            tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])

            for _,_,files in os.walk(src_dir):
                for file in files:
                    shutil.copyfile(os.path.join(src_dir,file), os.path.join(tar_dir, dir + '.' + file))
    print('{} had been moved.'.format(val_frame_4_path))

    for root,dirs,_ in os.walk(val_frame_output_path): 
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            for _,_,files in os.walk(src_dir):
                if len(files) == 0:
                    dpa.log_write('{} do not have any face'.format(dir), val_log_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--face_dir', type=str,
        help='The location of output face features set.', default='./datasets/IQIYIdataset0')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
