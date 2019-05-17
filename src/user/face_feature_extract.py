# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
# import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import math

embeddings_path = './logs/face_feats.txt'
face_dists0_path = './logs/face_dists0.txt'
face_dists1_path = './logs/face_dists1.txt'

def main(args):
    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            paths = get_dir(args.aligned_face_dir)

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
            last_batch = nrof_images % args.batch_size
            emb_array = np.zeros((nrof_images, embedding_size))
            lab_array = np.zeros((nrof_images,))
            for i in range(nrof_batches):
                feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.batch_size}
                emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
                lab_array[lab] = lab
                emb_array[lab, :] = emb
                if i % 10 == 9:
                    print('.', end='')
                    sys.stdout.flush()

            feed_dict = {phase_train_placeholder:False, batch_size_placeholder:last_batch}
            emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
            lab_array[lab] = lab
            emb_array[lab, :] = emb

            print('')
            embeddings = np.zeros((nrof_images, embedding_size))
            embeddings = emb_array

            embeddings0 = np.tile(embeddings[0],(len(embeddings),1))
            face_dists1 = facenet.distance(embeddings0, embeddings, distance_metric = 1)
            face_dists0 = facenet.distance(embeddings0, embeddings, distance_metric = 0)
            # face_dists = 1 - np.dot(embeddings, embeddings[2]) / \
            #             (np.linalg.norm(embeddings[2])*np.linalg.norm(embeddings, axis = 1))
            # face_dists1 = np.arccos(face_dists) / math.pi
            np.savetxt(face_dists0_path, face_dists0)
            np.savetxt(face_dists1_path, face_dists1)
            np.savetxt(embeddings_path, embeddings)
            print(face_dists0)
            print(face_dists1)

def get_dir(path):      #获取目录路径
    paths = []
    for root,dirs,_ in os.walk(path):     #遍历path,进入每个目录都调用visit函数
        # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        dirs.sort()
        for dir in dirs:
            tmp_dir = os.path.join(root,dir)
            for _,_,files in os.walk(tmp_dir):
                for file in files:
                    if file.split('.')[-1] == 'png':
                        paths.append(os.path.join(tmp_dir,file))      #把目录和文件名合成一个路径
                        labels.append(tmp_dir.split('/')[-1])
    return paths, labels
  
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

    # parser.add_argument('--lfw_pairs', type=str,
    #     help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    # parser.add_argument('--lfw_nrof_folds', type=int,
    #     help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    # parser.add_argument('--distance_metric', type=int,
    #     help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    # parser.add_argument('--use_flipped_images', 
    #     help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    # parser.add_argument('--subtract_mean', 
    #     help='Subtract feature mean before calculating distance.', action='store_true')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

# python face_feature_extract.py \
# ./datasets/IQIYIdataset/train_frame_1 \
# ./models/facenet/20180402-114759 \
# --use_fixed_image_standardization


## 减去特征均值？翻转图片？
