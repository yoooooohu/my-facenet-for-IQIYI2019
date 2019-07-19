# -*- coding: utf-8 -*-
######################################################
# read offical pickle files.
import pdb
import sys
import numpy as np
import pickle
import time

if sys.version_info >= (3, ):   # acquire py version
    import pickle
    def load_pickle(fin):
        return pickle.load(fin, encoding='bytes')
else:
    import cPickle as pickle
    def load_pickle(fin):
        return pickle.load(fin)

# read pickle
def load_IQIYI_pickle_data(path):
    print('loading "{}"...'.format(path))
    with open(path, 'rb') as fin:
        return load_pickle(fin)

# [frame_str, bbox, det_score, quality_score, feat] for face_feat
# [frame_str, bbox, det_score, feat] for head_feat
# [frame_str, bbox, feat] for body_feat
# feat for audio_feat

# Extract face features of a specified video
def read_face_features(train_feats_dict, video_name, face_qua_threshold = 50):
    fece_feats = train_feats_dict[video_name]
    fece_feat_dict = []
    for _, fece_feat in enumerate(fece_feats):
        [frame_str, bbox, det_score, quality_score, feat] = fece_feat
        if quality_score >= face_qua_threshold:
            fece_feat_dict.append(feat)
    return fece_feat_dict

# Extract head features of a specified video
def read_head_features(train_feats_dict, video_name):
    head_feats = train_feats_dict[video_name]
    head_feat_dict = []
    for _, head_feat in enumerate(head_feats):
        [frame_str, bbox, det_score, feat] = head_feat
        head_feat_dict.append(feat)
    return head_feat_dict    

# Extract body features of a specified video
def read_body_features(train_feats_dict, video_name):
    body_feats = train_feats_dict[video_name]
    body_feat_dict = []
    for _, body_feat in enumerate(body_feats):
        [frame_str, bbox, feat] = body_feat
        body_feat_dict.append(feat)
    return body_feat_dict  

# Extract audio features of a specified video
def read_audio_features(train_feats_dict, video_name):
    return train_feats_dict[video_name]




def get_val_image_labels(label_path):
    def by_name(d):
        return int(d[1].split('_')[-1].split('.')[0])

    result_dict = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(" ")
            for name in data[1:]:
                result_dict.update({name.split('.')[0]: data[0]})
        return result_dict

def get_train_image_labels(label_path):
    def by_name(d):
        return int(d[1].split('_')[-1].split('.')[0])

    result_dict = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(" ")
            result_dict.update({data[0].split('.')[0]: data[1]})
    return result_dict


def save_variable(var, filename):
    f=open(filename,'wb')
    pickle.dump(var,f)
    print("{} file is updated!".format(filename))
    f.close()
 
def load_variable(filename):
    f=open(filename,'rb')
    var=pickle.load(f)
    print("{} file  is loaded!".format(filename))
    f.close()
    return var

def log_write(message, filename):
    f=open(filename,'a+')
    f.writelines(message + '      ' + time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())) + '\n')
    print('log: ' + message)
    f.close()

def line_write(message, filename):
    f=open(filename,'a+')
    f.writelines(message + '\n')
    f.close()

def calculate_mAP(gt_val_path, my_val_path):
    # store the including video names per label in gt_val
    id2videos = dict()
    with open(gt_val_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split(' ')
            id2videos[terms[0]] = terms[1:]
    # num of labels
    id_num = len(lines)

    # store the including video names per label in my_gt_val
    my_id2videos = dict()
    with open(my_val_path, 'r') as fin:
        lines = fin.readlines()
        assert(len(lines) <= id_num)
        for line in lines:
            terms = line.strip().split(' ')
            tmp_list = []
            for video in terms[1:]:
                if video not in tmp_list:
                    tmp_list.append(video)
            my_id2videos[terms[0]] = tmp_list

    ap_total = 0.
    for cid in id2videos:
        videos = id2videos[cid]
        if cid not in my_id2videos:
            continue
        my_videos = my_id2videos[cid]
        # recall number upper bound
        assert(len(my_videos) <= 100)
        ap = 0.
        ind = 0.
        for ind_video, my_video in enumerate(my_videos):
            if my_video in videos:
                ind += 1        # 查准个数
                ap += ind / (ind_video + 1)     # p: precision，预测正确的个数/预测总个数
        ap_total += ap / len(videos)

    return ap_total / id_num

def calculate_Acc(gt_val_path, my_val_path):
    gt_dict = get_val_image_labels(gt_val_path)
    # store the including video names per label in gt_val
    id2videos = dict()
    with open(gt_val_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split(' ')
            id2videos[terms[0]] = terms[1:]
    # num of labels
    id_num = len(lines)

    # store the including video names per label in my_gt_val
    my_id2videos = dict()
    with open(my_val_path, 'r') as fin:
        lines = fin.readlines()
        assert(len(lines) <= id_num)
        for line in lines:
            terms = line.strip().split(' ')
            tmp_list = []
            for video in terms[1:]:
                if video not in tmp_list:
                    tmp_list.append(video)
            my_id2videos[terms[0]] = tmp_list

    error_videos = []
    right_num = 0.
    sum_num = 0.
    for cid in my_id2videos:
        my_videos = my_id2videos[cid]
        if cid not in id2videos:
            for _, my_video in my_videos:
                tmp = my_video.split('.')[0]
                if tmp in gt_dict.keys():
                    error_videos.append([my_video, cid, gt_dict[tmp]])
                else:
                    error_videos.append([my_video, cid, 0])
                sum_num += 1
        else:
            videos = id2videos[cid]
            # recall number upper bound
            # assert(len(my_videos) <= 100)
            for _, my_video in enumerate(my_videos):
                if my_video in videos:
                    right_num += 1
                else:
                    tmp = my_video.split('.')[0]
                    if tmp in gt_dict.keys():
                        error_videos.append([my_video, cid, gt_dict[tmp]])
                    else:
                        error_videos.append([my_video, cid, 0])
                sum_num += 1
    
    return right_num / sum_num, error_videos






# 读取所提取的视频帧的详细数据信息
# [frame_str, bbox, det_score, quality_score, feat] for face_feat
    # [x1, y1, x2, y2] = bbox
    # assert(0<=x1<=x2)
    # assert(0<=y1<=y2)
    # assert(type(det_score)==float)
    # assert(type(quality_score)==float)
    # assert(feat.dtype==np.float16 and feat.shape[0]==512)
# [frame_str, bbox, det_score, feat] for head_feat
# [frame_str, bbox, feat] for body_feat
# audio_feat for audio_feat
#
#
# def read_feats_dict(feat_name, feats_dict, video_name, type):
#     feats = feats_dict[video_name]
#     feat_dict = []
#     for _, feat in enumerate(feats):
#         if type == "face":
#             [frame_str, bbox, det_score, quality_score, feat] = feat
#             if feat_name == "frame_str":
#                 feat_dict.append(frame_str)
#             else if feat_name == "bbox":
#                 feat_dict.append(bbox)
#             else if feat_name == "det_score":
#                 feat_dict.append(det_score)
#             else if feat_name == "quality_score":
#                 feat_dict.append(quality_score)  
#             else if feat_name == "feat":
#                 feat_dict.append(feat) 
#             else:
#                 print("This feat_name is wrong input.")
#                 print("It should be [frame_str, bbox, det_score, quality_score, feat] for face_feat")

#         else if type == "head":
#             [frame_str, bbox, det_score, feat] = feat
#             if feat_name == "frame_str":
#                 feat_dict.append(frame_str)
#             else if feat_name == "bbox":
#                 feat_dict.append(bbox)
#             else if feat_name == "det_score":
#                 feat_dict.append(det_score)
#             else if feat_name == "feat":
#                 feat_dict.append(feat)
#             else:
#                 print("This feat_name is wrong input.")
#                 print("It should be [frame_str, bbox, det_score, feat] for head_feat")

#         else if type == "body":
#             [frame_str, bbox, feat] = feat
#             if feat_name == "frame_str":
#                 feat_dict.append(frame_str)
#             else if feat_name == "bbox":
#                 feat_dict.append(bbox)
#             else if feat_name == "feat":
#                 feat_dict.append(feat)
#             else:
#                 print("This feat_name is wrong input.")
#                 print("It should be [frame_str, bbox, feat] for body_feat")

#         else if type == "audio":
#             audio_feat = feat
#             if feat_name == "audio_feat":
#                 feat_dict.append(audio_feat)
#             else:
#                 print("This feat_name is wrong input.")
#                 print("It should be audio_feat for audio_feat")
#         else:
#             print("This type is wrong input.")

#     return feat_dict