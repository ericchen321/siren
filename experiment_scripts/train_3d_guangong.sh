python experiment_scripts/train_sdf.py \
--batch_size 8192 \
--num_epochs 401 \
--model_type sine \
--point_cloud_path data/our_shapes/guangong.xyz \
--experiment_name guangong \
--epochs_til_ckpt 100 \
--steps_til_summary 5000
