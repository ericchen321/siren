python experiment_scripts/train_sdf.py \
--batch_size 4096 \
--num_epochs 400 \
--model_type sine \
--point_cloud_path data/our_shapes/beard_man.xyz \
--experiment_name beard_man \
--epochs_til_ckpt 100 \
--steps_til_summary 5000
