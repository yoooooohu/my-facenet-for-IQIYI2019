# -*- coding: utf-8 -*-
# command:
# python src/user/faceFeat_calculation.py
import data_processing_AIQIYI as dpa
import numpy as np
import sys
import os
import argparse

# ground truth paths
train_gt_path = './offical_files/train_gt.txt'
# log record
log_loc = './generate_files/log.txt'

miss_feat_label_path = './generate_files/miss_train_feat_label.txt'
# data record
unnamed_face_video_path = './generate_files/unnamed_train_face_video.txt'
# stored pickle
face_feat_path = './generate_files/my_train_face_per_video.pickle'
feature_face_path = './generate_files/train_feature_face.pickle'

def main(args):
	gt_dict = dpa.get_train_image_labels(train_gt_path)

	# generate the feature vector merge dict 
	face_feature_dict = {}

	for n in range(0, 10034):
		face_feature_dict.update({'{}'.format(n + 1): []})

	# provisional feat lib
	perindex_face_feat_lib = np.zeros((10034, 512))

	face_feats_dict = dpa.load_IQIYI_pickle_data(face_feat_path)
	cnt = 0
	cnt_all = len(face_feats_dict)
	print(cnt_all)
	for _, video_name in enumerate(face_feats_dict):

		cnt = cnt + 1
		if cnt % 1000 == 0:
			print(" face video {} / {}".format(cnt, cnt_all))

		# extract all feat and classify them by index
		face_feat = face_feats_dict[video_name]

		if np.isnan(face_feat).any():
			dpa.log_write("the video {} don't have face data".format(video_name), log_loc)
			dpa.line_write(video_name, unnamed_face_video_path)			
		else:
			face_feature_dict[gt_dict[video_name]].append(face_feat)

	# mean the all face feature vectors per person
	for n in range(0, 10034):	# 10034
		index = '{}'.format(n + 1)
		if len(face_feature_dict[index]):
			perindex_face_feat_lib[n] = np.mean(np.array(face_feature_dict[index]), axis = 0)
		else:
			# the miss feat will be set all 5 so that it can not be choose
			dpa.log_write("this train face sets miss label -> " + index, log_loc)
			dpa.line_write("face" + index, miss_feat_label_path)

	del face_feature_dict
	dpa.save_variable(perindex_face_feat_lib, feature_face_path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))