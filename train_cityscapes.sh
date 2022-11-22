python train.py --name oasis_cityscapes_unsupervised_gen_8_bilinear_residual_nearest_nonresidual_wavelet_spade --dataset_mode cityscapes --gpu_ids 0 \
--dataroot /data/public/cityscapes --no_labelmix \
--batch_size 1 --model_supervision 0 --supervised_num 20 \
--Du_patch_size 64 --netDu wavelet  \
--netG 8 --channels_G 16 \
 --num_epochs 500