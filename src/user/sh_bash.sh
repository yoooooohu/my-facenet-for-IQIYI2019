# 注意几点：
# 1.用的什么MTCNN的人脸数据集
# 2.用的什么模型
# 3.用哪些gpu
# 4.生成/使用文件
# add facenet source to python path
export PYTHONPATH=/home/hyp/facenet/src:$PYTHONPATH;
# extract train face img in videos
nohup python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_1 \
./datasets/IQIYIdataset/train_frame_1 \
--image_size 160 \
--margin 32 \
--random_order \
--user \
--gpu_memory_fraction 0.08 \
--gpu_device 3 \
--threshold_P 0.4 \
--threshold_R 0.7 \
--threshold_O 0.8 >> mtcnn_t1.out &
nohup python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_2 \
./datasets/IQIYIdataset/train_frame_2 \
--image_size 160 \
--margin 32 \
--random_order \
--user \
--gpu_memory_fraction 0.08 \
--gpu_device 3 \
--threshold_P 0.4 \
--threshold_R 0.7 \
--threshold_O 0.8 >> mtcnn_t2.out &
nohup python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_3 \
./datasets/IQIYIdataset/train_frame_3 \
--image_size 160 \
--margin 32 \
--random_order \
--user \
--gpu_memory_fraction 0.08 \
--gpu_device 3 \
--threshold_P 0.4 \
--threshold_R 0.7 \
--threshold_O 0.8 >> mtcnn_t3.out &
# extract val face img in videos
nohup python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_1 \
./datasets/IQIYIdataset/val_frame_1 \
--image_size 160 \
--margin 32 \
--random_order \
--user \
--gpu_memory_fraction 0.08 \
--gpu_device 3 \
--threshold_P 0.4 \
--threshold_R 0.7 \
--threshold_O 0.8 >> mtcnn_v1.out &
nohup python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_2 \
./datasets/IQIYIdataset/val_frame_2 \
--image_size 160 \
--margin 32 \
--random_order \
--user \
--gpu_memory_fraction 0.08 \
--gpu_device 3 \
--threshold_P 0.4 \
--threshold_R 0.7 \
--threshold_O 0.8 >> mtcnn_v2.out &
nohup python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_3 \
./datasets/IQIYIdataset/val_frame_3 \
--image_size 160 \
--margin 32 \
--random_order \
--user \
--gpu_memory_fraction 0.08 \
--gpu_device 3 \
--threshold_P 0.4 \
--threshold_R 0.7 \
--threshold_O 0.8 >> mtcnn_v3.out &
nohup python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_4 \
./datasets/IQIYIdataset/val_frame_4 \
--image_size 160 \
--margin 32 \
--random_order \
--user \
--gpu_memory_fraction 0.08 \
--gpu_device 3 \
--threshold_P 0.4 \
--threshold_R 0.7 \
--threshold_O 0.8 >> mtcnn_v4.out &
######################################################->
# merge the set by mean of labels
# train data generation
python src/user/dataset_generation.py \
--face_dir ./datasets/IQIYIdataset;
# val data generation
python src/user/dataset_generation.py \
--generate_val_pairs_dir ./data/IQIYI_pairs_new.txt \
--face_dir ./datasets/IQIYIdataset \
--val_mode;
######################################################
# pertrain by tripleloss from offical model
nohup python src/train_tripletloss.py \
--logs_base_dir ./logs/facenet/ \
--models_base_dir ./models/facenet/ \
--data_dir ./datasets/IQIYIdataset478/train_frame/ \
--pretrained_model ./models/facenet/20180402-114759/model-20180402-114759.ckpt-275 \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir ./datasets/IQIYIdataset478/val_frame/ \
--lfw_pairs ./data/IQIYI_pairs.txt \
--embedding_size 512 \
--batch_size 150 \
--optimizer RMSPROP \
--learning_rate 0.002 \
--weight_decay 1e-4 \
--gpu_device 2 \
--max_nrof_epochs 500 >> train_triplet.out &
######################################################
# extract face features per video
# train feat extract
python src/user/face_feature_extract.py \
./datasets/IQIYIdataset \
./models/facenet/20180402-114759 \
--use_fixed_image_standardization \
--gpu_device 3 \
--face_feat_dir ./generate_files/train_feature_face.pickle;
# extract face features per video
# val feat extract
python src/user/face_feature_extract.py \
./datasets/IQIYIdataset \
./models/facenet/20180402-114759 \
--use_fixed_image_standardization \
--face_feat_dir ./generate_files/my_val_face_per_video.pickle \
--gpu_device 1 \
--val_mode;
# calculation feat per label
nohup python src/user/final_validation.py \
--face_feat_dir ./generate_files/my_val_face_per_video.pickle \
--feature_face_per_label_dir ./generate_files/train_feature_face.pickle \
--subtract_mean  >> result.out &





python src/user/video_validation.py \
--face_feat_dir ./generate_files/my_val_face_per_video.pickle \
--feature_face_per_label_dir ./generate_files/train_feature_face.pickle \
--subtract_mean \
--video_name IQIYI_VID_VAL_0191299