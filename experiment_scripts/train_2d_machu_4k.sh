python experiment_scripts/train_img.py \
--experiment_name machu_4k \
--model_type=sine \
--image_path data/machu_4k_gt.jpg \
--num_epochs 50001 \
--point_batch_size 1048576 \
--epochs_til_ckpt 100 \
--steps_til_summary 1000