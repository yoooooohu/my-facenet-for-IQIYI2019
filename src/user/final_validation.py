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
	val_classificaion_dict_path = './generate_files/val_classificaion_dict{}_{}.pickle'.format(args.euclidean_dist, args.subtract_mean)
	
	if 'val' in args.running_mode:
		print('-------------------- validation --------------------')

		# generate the feature vector merge dict 
		video_face_output_dict = {}
		for n in range(0, 10034):
			video_face_output_dict.update({'{}'.format(n + 1): []})

		face_feats_dict = dpa.load_IQIYI_pickle_data(args.face_feat_dir)
		perindex_face_feat_lib = dpa.load_variable(args.feature_face_per_label_dir)

		if args.subtract_mean:
			sub_mean = np.mean(perindex_face_feat_lib, axis=0)
		else:
			sub_mean = 0.0

		cnt = 0
		cnt_all = len(face_feats_dict)
		print('The number of val videos is {}.'.format(cnt_all))
		for _, video_name in enumerate(face_feats_dict):		
			cnt = cnt + 1
			if cnt % 500 == 0:
				print(" face video {} / {}".format(cnt, cnt_all))

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
				
				min_dist = np.nanmin(face_dists)	# 最小值时忽略nan

				if min_dist < 1: # args.limit_max_dist:
					perfect_ind = np.nanargmin(face_dists)
					cal_ind = '{}'.format(perfect_ind + 1)
					video_face_output_dict[cal_ind].append([min_dist, video_name])

		dpa.save_variable(video_face_output_dict, val_classificaion_dict_path)

	if 'map' in args.running_mode:
		# forecast & mAP computing prosess
		video_face_output_dict = dpa.load_variable(val_classificaion_dict_path)

		my_val_f = open(my_val_gt_path, 'wb')
		most_cnt = 0
		for n in range(0, 10034):
			blank_str = '{}'.format(n + 1)
			# becouse of the mAP need to be sorted by confident rate
			sorted_dist = video_face_output_dict[blank_str]

			sorted_dist.sort()
			
			dist_cnt = len(sorted_dist)

			if most_cnt < dist_cnt:
				most_cnt = dist_cnt
			len_cnt = 0
			for _,video_name in sorted_dist:
				len_cnt += 1
				if len_cnt > 100:
					break
				blank_str += ' {}.mp4'.format(video_name)
			my_val_f.writelines(blank_str + '\n')


		print("generate '{}' file".format(my_val_gt_path))
		my_val_f.close()

		# get Acc & mAP
		mAP = dpa.calculate_mAP(val_gt_path, my_val_gt_path)
		Acc, error_videos_list = dpa.calculate_Acc(val_gt_path, my_val_gt_path)

		dpa.log_write("euclidean_dist {}-subtract_mean {} mAP: {} Acc: {} most cnt: {}"\
									.format(args.euclidean_dist, args.subtract_mean, mAP, Acc, most_cnt), result_loc)

		my_el_f = open(wrong_videos_path,'wb')
		for error_video in error_videos_list:
			my_el_f.writelines('{} {} {}\n'.format(error_video[0], error_video[1], error_video[2]))		
		print("generate '{}' file".format(wrong_videos_path))
		my_el_f.close()




def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--euclidean_dist',
        help='default : cosine distance. if def : Euclidean distance.', action='store_true')
    parser.add_argument('--limit_max_dist', type=float,
        help='Distance Upper bound. Suggestion: 0.35 for cosine; 1 for Euclidean.', default=0.35)
    parser.add_argument('--subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--running_mode', type=str, help='running_mode.', default='val_map')
    parser.add_argument('--face_feat_dir', type=str,
        help='The location of all val video.', default='./generate_files/my_val_face_per_video.pickle')
    parser.add_argument('--feature_face_per_label_dir', type=str,
        help='The location of all labels feats.', default='./generate_files/train_feature_face.pickle')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    