
# start_cp=checkpoint_1103_160
# all_splits=($start_cp)
# # all_splits+=('checkpoint_1098_40')
# # all_splits+=('checkpoint_1098_50')
# # all_splits+=('checkpoint_1099_60')
# # all_splits+=('checkpoint_1099_70')
# # all_splits+=('checkpoint_1100_80')
# # all_splits+=('checkpoint_1100_90')
# # all_splits+=('checkpoint_1100_100')
# # all_splits+=('checkpoint_1101_110')
# # all_splits+=('checkpoint_1101_120')
# # all_splits+=('checkpoint1101')
# # all_splits+=('checkpoint_1102_140')
# # all_splits+=('checkpoint_1102_150')
# # all_splits+=('checkpoint_1103_160')


# echo $all_splits

# for curr_cp in $all_splits; do
#     python dump_boundary.py "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats/top_1024/km_dir_1024/" --ckpt-path /nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats/top_1024/km_dir_1024/models_policy_noskip_try_debug_32/${curr_cp}.pt --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_debug_2/speecht5" --feat-name check_policy_gradient_${curr_cp} --split "speech_valid|text_valid"
# done

# DATA_DIR=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_clsattnkmeans_encoder_only_retry_2024_2l_diff_bnd_retry_small_batch/checkpoint_best.pt

# DATA_DIR=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gradseg/top_1024/km_dir_1024/
# CKPT_PATH=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gradseg/top_1024/km_dir_1024/model_codebook_0.3_clsattn_kmeans_encoder_only_retry_2024_2l_ablation_diff_bnd/checkpoint_best.pt

# DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/"
# CKPT_PATH=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattn_kmeans_encoder_only_retry_2024_2l_large_batch_diff_bnd_no_codebook_prob_large_reg/checkpoint_best.pt


# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true"


# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_train_clsattnbndkmeans|text_train"

# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_clsattnbndkmeans|text_valid"



# DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/"
# CKPT_PATH=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattn_kmeans_encoder_only_retry_2024_2l_large_batch_diff_bnd_no_codebook_prob_small_reg/checkpoint_best.pt


# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true"


# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_train_clsattnbndkmeans|text_train"

# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_clsattnbndkmeans|text_valid"





# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_clsattnkmeans_encoder_only_retry_2024_2l_diff_bnd_retry_smaller/checkpoint_best.pt


# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_train_clsattnbndkmeans|text_train"

# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_clsattnbndkmeans|text_valid"

# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true"


# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_clsattnkmeans_encoder_only_retry_2024_2l_diff_bnd_retry_larger/checkpoint_best.pt


# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_train_clsattnbndkmeans|text_train"

# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_clsattnbndkmeans|text_valid"

# python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true"




DATA_DIR=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/
CKPT_PATH=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_wav2bnd_kmeans_encoder_only_retry_2024_2l_ablation_diff_bnd/checkpoint_best.pt


python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid_true|text_valid_true"


python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_train|text_train"

python dump_boundary.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_valid|text_valid"