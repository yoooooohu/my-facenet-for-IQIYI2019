# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
from tensorflow.python.ops import data_flow_ops
from scipy import interpolate
import math
import data_processing_AIQIYI as dpa
from tqdm import tqdm
from tqdm._tqdm import trange
from collections import OrderedDict # 有序字典

miss_embedding_label_path = './logs/user/miss_embedding_label.txt'

def main(args):
    train_dir = args.aligned_face_dir + '/train_frame'
    val_1_dir = args.aligned_face_dir + '/val_frame_1'
    val_2_dir = args.aligned_face_dir + '/val_frame_2'
    val_3_dir = args.aligned_face_dir + '/val_frame_3'
    val_4_dir = args.aligned_face_dir + '/val_frame_4'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            if args.val_mode:
                paths, labels = get_dir([val_1_dir, val_2_dir, val_3_dir, val_4_dir])
            else:
                paths, labels = get_dir([train_dir])
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            #创建一个先入先出队列
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            # 多值入队
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            # 将 input_queue 中的 (image, label, control) 元祖 dequeue 出来，根据 control 里的内容
            #   对 image 进行各种预处理，然后将处理后的 (image, label) 打包成真正输入 model 的 batch
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            nrof_images = len(paths)
            labels_array = np.expand_dims(np.arange(0,nrof_images),1)
            image_paths_array = np.expand_dims(np.array(paths),1)

            control_array = np.zeros_like(labels_array, np.int32)
            if args.use_fixed_image_standardization:
                control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION

            sess.run(eval_enqueue_op, {image_paths_placeholder: image_paths_array, \
                    labels_placeholder: labels_array, control_placeholder: control_array})

            embedding_size = int(embeddings.get_shape()[1])
            nrof_batches = nrof_images // args.batch_size
            emb_array = np.zeros((nrof_images, embedding_size))
            lab_array = np.zeros((nrof_images,))
            for i in tqdm(range(nrof_batches)):
                feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.batch_size}
                emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
                lab_array[lab] = lab
                emb_array[lab, :] = emb
                if i % 10 == 9:
                    print('.', end='')
                    sys.stdout.flush()
            # 无法成batch的必须要另行处理，否则error
            last_batch = nrof_images % args.batch_size

            if last_batch:
                feed_dict = {phase_train_placeholder:False, batch_size_placeholder:last_batch}
                emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
                lab_array[lab] = lab
                emb_array[lab, :] = emb

            print('')
            embeddings = np.zeros((nrof_images, embedding_size))
            embeddings = emb_array

            print('embeddings generated sucess')

            if args.val_mode:
                my_pickle = {}
                ind = 0
                for name, cnt in labels.items():    # labels为有序字典
                    # todo：是否可以利用聚类的方法进行一遍预筛选
                    mean_embeddings = np.mean(embeddings[ind:ind+cnt], axis = 0)
                    my_pickle.update({name: mean_embeddings})
                    ind = ind + cnt
                dpa.save_variable(my_pickle, args.face_feat_dir)
            else:
                perindex_face_feat_lib = np.zeros((10034, 512))
                ind = 0
                for name, cnt in labels.items():    # labels为有序字典
                    # todo：是否可以利用聚类的方法进行一遍预筛选
                    if cnt:
                        perindex_face_feat_lib[int(name) - 1] = np.mean(embeddings[ind:ind+cnt], axis = 0)
                        ind = ind + cnt
                    else:
                        dpa.log_write(name, miss_embedding_label_path)
                dpa.save_variable(perindex_face_feat_lib, args.face_feat_dir)
            
            print('{} updated sucess'.format(args.face_feat_dir))


            # face_dists0_path = './logs/face_dists0.txt'
            # face_dists1_path = './logs/face_dists1.txt'
            # embeddings_path = './logs/face_feats.txt'
            # embeddings0 = np.tile(embeddings[0],(len(embeddings),1))
            # face_dists1 = facenet.distance(embeddings0, embeddings, distance_metric = 1)
            # face_dists0 = facenet.distance(embeddings0, embeddings, distance_metric = 0)

            # np.savetxt(face_dists0_path, face_dists0)
            # np.savetxt(face_dists1_path, face_dists1)
            # np.savetxt(embeddings_path, embeddings)


#获取目录路径
def get_dir(paths):      
    img_paths = []
    img_labels = OrderedDict()  # 必须有序，否则label与embedding对应时会出大问题
    for path in paths:
        for root,dirs,_ in os.walk(path):     #遍历path,进入每个目录都调用visit函数
            # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
            dirs.sort()     # 必须排序，否则picture与label对应会失败
            for dir in dirs:
                tmp_dir = os.path.join(root,dir)
                for _,_,files in os.walk(tmp_dir):
                    img_labels.update({tmp_dir.split('/')[-1]: len(files)})
                    for file in files:
                        img_paths.append(os.path.join(tmp_dir,file))      #把目录和文件名合成一个路径
    return img_paths, img_labels
  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
        
    parser.add_argument('aligned_face_dir', type=str,
        help='Path to the data directory containing aligned input face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true') 
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch in the test set.', default=100)
    parser.add_argument('--face_feat_dir', type=str,
        help='The location of output face features set.', default='./generate_files/face_feature_extract.pickle')
    parser.add_argument('--val_mode', help='This is val mode.', action='store_true')
    parser.add_argument('--gpu_device', type=str, help='which cpu you want to utilize', default='0,1,2,3')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

