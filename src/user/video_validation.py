# -*- coding: utf-8 -*-
# command:
# python src/user/final_validation.py --euclidean_dist
# python src/user/final_validation.py --subtract_mean
# python src/user/final_validation.py --euclidean_dist --subtract_mean
# python src/user/final_validation.py
import data_processing_AIQIYI as dpa
import numpy as np
import sys
import os
import facenet
import argparse

# ground truth paths
val_gt_path = './offical_files/val_gt.txt'
train_gt_path = './offical_files/train_gt.txt'
my_val_gt_path = './generate_files/result.txt'		# 预测gt
# log record
log_loc = './logs/user/final_validation_log.txt'	# 
result_loc = './logs/user/validation_result.txt'	# 预测结果: mAP Acc等
# data record
unnamed_face_video_path = './generate_files/unlabeled_val_video.txt'	# 预测的无标签数据
wrong_videos_path = './generate_files/wrong_videos.txt'					# 预测错误的无标签数据


def main(args):
	video_name = args.video_name
	print('validating the video {}...'.format(video_name))

	face_feats_dict = dpa.load_IQIYI_pickle_data(args.face_feat_dir)
	perindex_face_feat_lib = dpa.load_variable(args.feature_face_per_label_dir)
	val_grand_truth = dpa.get_val_image_labels(val_gt_path)

	if args.subtract_mean:
		sub_mean = np.mean(perindex_face_feat_lib, axis=0)
	else:
		sub_mean = 0.0
	face_dists = np.zeros([10034, 1])
	# extract all feat and classify them by index
	face_feat = face_feats_dict[video_name]

	if np.isnan(face_feat).any():
		dpa.log_write("the val video {} don't have face data".format(video_name), log_loc)
		dpa.line_write(video_name, unnamed_face_video_path)	
	else:
		mean_face_feat_tile = np.tile(face_feat,(10034,1))

		if args.euclidean_dist:
			face_dists = facenet.distance(mean_face_feat_tile-sub_mean, perindex_face_feat_lib-sub_mean, distance_metric = 0)
		else:
			face_dists = facenet.distance(mean_face_feat_tile-sub_mean, perindex_face_feat_lib-sub_mean, distance_metric = 1)
		
		np.savetxt ('./logs/tmp/{}_dist.txt'.format(video_name), face_dists)

		min_dist = np.nanmin(face_dists)	# 最小值时忽略nan
		if video_name in val_grand_truth.keys():
			actual_label = val_grand_truth[video_name]
			actual_dist = face_dists[int(actual_label)-1]
		else:
			actual_label = 0
			actual_dist = 0

		if min_dist < 1: # args.limit_max_dist:
			perfect_ind = np.nanargmin(face_dists)
			print('prediction:{}-{} actually is {}-{}' \
				.format(perfect_ind + 1, min_dist, actual_label, actual_dist))
		else:
			print('prediction: miss label actually is {}-{}'.format(actual_label, actual_dist))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--euclidean_dist',
        help='default : cosine distance. if def : Euclidean distance.', action='store_true')
    parser.add_argument('--limit_max_dist', type=float,
        help='Distance Upper bound. Suggestion: 0.35 for cosine; 1 for Euclidean.', default=0.35)
    parser.add_argument('--subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--face_feat_dir', type=str,
        help='The location of all val video.', default='./generate_files/my_val_face_per_video.pickle')
    parser.add_argument('--feature_face_per_label_dir', type=str,
        help='The location of all labels feats.', default='./generate_files/train_feature_face.pickle')
    parser.add_argument('--video_name', type=str,
        help='which video you want to validate.', default='IQIYI_VID_VAL_0187008')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    