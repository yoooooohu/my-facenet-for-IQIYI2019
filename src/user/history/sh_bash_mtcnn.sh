python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_3 \
./datasets/IQIYIdataset677/train_frame_3 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 1 \
--threshold_P 0.6 \
--threshold_R 0.7 \
--threshold_O 0.7 \
--gpu_device 0,1;
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_2 \
./datasets/IQIYIdataset677/train_frame_2 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 1 \
--threshold_P 0.6 \
--threshold_R 0.7 \
--threshold_O 0.7 \
--gpu_device 0,1;
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_1 \
./datasets/IQIYIdataset677/train_frame_1 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 1 \
--threshold_P 0.6 \
--threshold_R 0.7 \
--threshold_O 0.7 \
--gpu_device 0,1;
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_1 \
./datasets/IQIYIdataset677/val_frame_1 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 1 \
--threshold_P 0.6 \
--threshold_R 0.7 \
--threshold_O 0.7 \
--gpu_device 0,1;
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_2 \
./datasets/IQIYIdataset677/val_frame_2 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 1 \
--threshold_P 0.6 \
--threshold_R 0.7 \
--threshold_O 0.7 \
--gpu_device 0,1;
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_3 \
./datasets/IQIYIdataset677/val_frame_3 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 1 \
--threshold_P 0.6 \
--threshold_R 0.7 \
--threshold_O 0.7 \
--gpu_device 0,1;
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_4 \
./datasets/IQIYIdataset677/val_frame_4 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 1 \
--threshold_P 0.6 \
--threshold_R 0.7 \
--threshold_O 0.7 \
--gpu_device 0,1;
