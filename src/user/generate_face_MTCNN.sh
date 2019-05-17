for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_1 \
./datasets/IQIYIdataset0/train_frame_1 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
--threshold_P 0.7 \
--threshold_R 0.8 \
--threshold_O 0.8 \
& done
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_2 \
./datasets/IQIYIdataset0/train_frame_2 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
--threshold_P 0.7 \
--threshold_R 0.8 \
--threshold_O 0.8 \
& done
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
/cdisk/wenfahu/20190408/IQIYIdataset/train_frame_3 \
./datasets/IQIYIdataset0/train_frame_3 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
--threshold_P 0.7 \
--threshold_R 0.8 \
--threshold_O 0.8 \
& done

for N in {1..4}; do \
python src/align/align_dataset_mtcnn_hyp_gpu.py \
/cdisk/wenfahu/20190408/IQIYIdataset/val_frame_1 \
./datasets/IQIYIdataset0/val_frame_1 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
--threshold_P 0.7 \
--threshold_R 0.8 \
--threshold_O 0.8 \
& done