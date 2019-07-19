# -*- coding: utf-8 -*-
# command: python src/user/train_dataset_generation.py

import data_processing_AIQIYI as dpa
import sys
import os
import shutil
import argparse
import random
import numpy as np

# ground truth paths
train_gt_path = './offical_files/train_gt.txt'
val_gt_path = './offical_files/val_gt.txt'

val_log_path = './logs/user/val_dataset_generation_log.txt'
train_log_path = './logs/user/train_dataset_generation_log.txt'

def main(args):
    if args.val_mode:
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

        cnt_list = np.zeros([10034,])

        for root,dirs,_ in os.walk(val_frame_1_path):     #遍历path,进入每个目录都调用visit函数
            # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
            for dir in dirs:
                if dir not in gt_dict.keys():
                    continue
                src_dir = os.path.join(root, dir)
                tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])
                gt = int(gt_dict[dir])

                for _,_,files in os.walk(src_dir):
                    for file in files:
                        cnt_list[gt-1] += 1
                        shutil.copyfile(os.path.join(src_dir,file), 
                            os.path.join(tar_dir, '%d_%04d.%s'%(gt, cnt_list[gt-1], file.split('.')[-1])))
        print('{} had been moved.'.format(val_frame_1_path))

        for root,dirs,_ in os.walk(val_frame_2_path):     #遍历path,进入每个目录都调用visit函数
            # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
            for dir in dirs:
                if dir not in gt_dict.keys():
                    continue            
                src_dir = os.path.join(root, dir)
                tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])
                gt = int(gt_dict[dir])

                for _,_,files in os.walk(src_dir):
                    for file in files:
                        cnt_list[gt-1] += 1
                        shutil.copyfile(os.path.join(src_dir,file), 
                            os.path.join(tar_dir, '%d_%04d.%s'%(gt, cnt_list[gt-1], file.split('.')[-1])))
        print('{} had been moved.'.format(val_frame_2_path))

        for root,dirs,_ in os.walk(val_frame_3_path):     #遍历path,进入每个目录都调用visit函数
            # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
            for dir in dirs:
                if dir not in gt_dict.keys():
                    continue            
                src_dir = os.path.join(root, dir)
                tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])
                gt = int(gt_dict[dir])

                for _,_,files in os.walk(src_dir):
                    for file in files:
                        cnt_list[gt-1] += 1
                        shutil.copyfile(os.path.join(src_dir,file), 
                            os.path.join(tar_dir, '%d_%04d.%s'%(gt, cnt_list[gt-1], file.split('.')[-1])))
        print('{} had been moved.'.format(val_frame_3_path))

        for root,dirs,_ in os.walk(val_frame_4_path):     #遍历path,进入每个目录都调用visit函数
            # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
            for dir in dirs:
                if dir not in gt_dict.keys():
                    continue            
                src_dir = os.path.join(root, dir)
                tar_dir = os.path.join(val_frame_output_path, gt_dict[dir])
                gt = int(gt_dict[dir])

                for _,_,files in os.walk(src_dir):
                    for file in files:
                        cnt_list[gt-1] += 1
                        shutil.copyfile(os.path.join(src_dir,file), 
                            os.path.join(tar_dir, '%d_%04d.%s'%(gt, cnt_list[gt-1], file.split('.')[-1])))
        print('{} had been moved.'.format(val_frame_4_path))

        miss_data_list = []
        for root,dirs,_ in os.walk(val_frame_output_path): 
            for dir in dirs:
                src_dir = os.path.join(root, dir)
                for _,_,files in os.walk(src_dir):
                    if len(files) == 0:
                        dpa.log_write('{} do not have any face'.format(dir), val_log_path)
                        miss_data_list.append(int(dir))
    
        if args.generate_val_pairs_dir:
            f=open(args.generate_val_pairs_dir, 'wb+')
            f.writelines('10 300\n')

            for line_cnt in range(6000):
                if (line_cnt % 600) < 300:
                    rand_dir = random.randint(1, 10034)
                    while rand_dir in miss_data_list:
                        rand_dir = random.randint(1, 10034)
                    file1 = ''
                    file2 = '' 
                    for _,_,files in os.walk(os.path.join(val_frame_output_path, str(rand_dir))):
                        files_cnt = len(files)
                        file1 = random.randint(1, files_cnt)
                        file2 = random.randint(1, files_cnt)
                        # while file2 == file1:   # 可能存在一个数据集只有一个数据的情况！
                        #     file2 = random.randint(1, files_cnt)                         

                    f.writelines('{} {} {}\n'.format(rand_dir, file1, file2))
                    print('{} {} {}'.format(rand_dir, file1, file2))
                else:
                    rand_dir1 = random.randint(1, 10034)
                    rand_dir2 = random.randint(1, 10034)
                    while rand_dir1 in miss_data_list:
                        rand_dir1 = random.randint(1, 10034)
                    while (rand_dir2 in miss_data_list) or (rand_dir1 == rand_dir2):
                        rand_dir2 = random.randint(1, 10034)
                    file1 = ''
                    file2 = ''
                    for _,_,files in os.walk(os.path.join(val_frame_output_path, str(rand_dir1))):
                        files_cnt = len(files)
                        file1 = random.randint(1, files_cnt)
                    for _,_,files in os.walk(os.path.join(val_frame_output_path, str(rand_dir2))):
                        files_cnt = len(files)
                        file2 = random.randint(1, files_cnt)              

                    f.writelines('{} {} {} {}\n'.format(rand_dir1, file1, rand_dir2, file2))
                    print('{} {} {} {}'.format(rand_dir1, file1, rand_dir2, file2))
                
            f.close()
            print('file {} had been created.'.format(args.generate_val_pairs_dir))



    else:
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
                        dpa.log_write('{} do not have any face'.format(dir), train_log_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--face_dir', type=str,
        help='The location of output face features set.', default='./datasets/IQIYIdataset0')
    parser.add_argument('--val_mode', help='If def, generate val datasets.', action='store_true')
    parser.add_argument('--generate_val_pairs_dir', type=str, help='If def, generate val pairs.', default='')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
