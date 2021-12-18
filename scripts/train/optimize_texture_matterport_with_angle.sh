python -m model.optimize --gpus 1 \
--root_path path/to/datasets/Matterport3D --dataset matterport \
--resize_size 256 --texture_size 4096,4096 \
--min_images 1 --max_images 1000 --scene 17DRP5sb8fy --matterport_region_index 0 \
--hierarchical --hierarchical_layers 4 \
--loss_weight content=7e1 \
--loss_weight style=1e-4 --style_weights="1000,1000,10,10,1000" \
--loss_weight tex_reg=5e3 \
--vgg_gatys_model_path path/to/models/vgg_conv.pth \
--renderer_mipmap path/to/git/neural-rendering-style-transfer/scripts/matterport/render_uv/build/matterport_renderer \
--learning_rate 1 --decay_step_size 3 \
--log_images_nth 5000 --batch_size 1 \
--max_epochs 7 \
--train_split 0.99 --val_split 0.01 \
--sampler_mode repeat --index_repeat 100 \
--save_texture --split_mode sequential \
--num_workers 4 \
--style_image_path path/to/datasets/styles/custom_21styles/14-2.jpg \
--style_pyramid_mode "multi" \
--gram_mode "current" \
--angle_threshold 40 \
--pyramid_levels 1 \
--min_pyramid_depth 0.2 \
--min_pyramid_height 256 \
--no_depth_scaling