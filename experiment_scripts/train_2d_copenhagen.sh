python experiment_scripts/train_img.py \
--experiment_name copenhagen \
--model_type=sine \
--image_path data/copenhagen.jpg \
--num_epochs 50000 \
--point_batch_size 1048576 \
--epochs_til_ckpt 1 \
--steps_til_summary 1000