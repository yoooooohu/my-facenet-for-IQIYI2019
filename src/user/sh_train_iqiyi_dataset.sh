python src/train_softmax.py \
--logs_base_dir ./logs/facenet/ \
--models_base_dir ./models/facenet/ \
--data_dir ./datasets/IQIYIdataset0/train_frame/ \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir ./datasets/lfw/lfw_mtcnnpy_160/ \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 75 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--gpu_device 2,3 \
--prelogits_norm_loss_factor 5e-4
# pertrain by softmax
python src/user/my_train_softmax.py \
--logs_base_dir ./logs/facenet/ \
--models_base_dir ./models/facenet/ \
--data_dir ./datasets/IQIYIdataset0/train_frame/ \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--pretrained_model ./models/facenet/20180402-114759/model-20180402-114759.ckpt-275 \
--lfw_dir ./datasets/lfw/lfw_mtcnnpy_160/ \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 75 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file ./data/learning_rate_schedule_classifier_pretrain.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.01 \
--validate_every_n_epochs 5 \
--gpu_device 3 \
--prelogits_norm_loss_factor 5e-4
# pertrain by tripleloss from offical model ############################
python src/train_tripletloss.py \
--logs_base_dir ./logs/facenet/ \
--models_base_dir ./models/facenet/ \
--data_dir ./datasets/IQIYIdataset0/train_frame/ \
--pretrained_model ./models/facenet/20180402-114759/model-20180402-114759.ckpt-275 \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir ./datasets/IQIYIdataset677/val_frame/ \
--lfw_pairs ./data/IQIYI_pairs.txt \
--embedding_size 512 \
--optimizer RMSPROP \
--learning_rate 0.002 \
--weight_decay 1e-4 \
--gpu_device 3 \
--max_nrof_epochs 500

##################################################################
# IQIYI val
--lfw_dir ./datasets/IQIYIdataset677/val_frame/ \
--lfw_pairs IQIYI_pairs.txt \
##################################################################

