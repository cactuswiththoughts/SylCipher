USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5

cd /nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024

# DATA_DIR=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/
# CKPT_PATH_BASE=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/model_codebook_0.3_orikmeans_encoder_only_retry_2024_2l
DATA_DIR=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd_encoder_only_e2e_0.63tokenf1_w2b/top_1024/km_dir_1024/
CKPT_PATH_BASE=/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd_encoder_only_e2e_0.63tokenf1_w2b/top_1024/km_dir_1024/model_codebook_0.3_orikmeans_encoder_only_retry_2024_2l

CKPT_PATH=$CKPT_PATH_BASE/checkpoint_best.pt

echo "dumping VALID_TRUE"
python dump_embedding_npy.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true" --postfix "valid_true" --use-discrete-labels 1 --tgt-enc-layer 0

echo "dumping TRAIN"
python dump_embedding_npy.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train_clsattnbndkmeans|text_train" --postfix "train" --use-discrete-labels 1 --tgt-enc-layer 0


echo "dumping VALID_TRUE"
python dump_embedding_npy.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true" --postfix "valid_true" --use-discrete-labels 1 --tgt-enc-layer 1

echo "dumping TRAIN"
python dump_embedding_npy.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train_clsattnbndkmeans|text_train" --postfix "train" --use-discrete-labels 1 --tgt-enc-layer 1


bash run_clus_speech_text.sh 1 0 $CKPT_PATH_BASE/ $CKPT_PATH_BASE/labels_layer_0 $CKPT_PATH_BASE/km_model_layer_0 1024


bash run_clus_speech_text.sh 1 1 $CKPT_PATH_BASE/ $CKPT_PATH_BASE/labels_layer_1 $CKPT_PATH_BASE/km_model_layer_1 1024


python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 0 "text" "speech"

python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 1 "text" "speech"

python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 0 "speech" "speech"

python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 1 "speech" "speech"

# python calculate_clus_metrics.py $CKPT_PATH_BASE "valid_true" 0 "text" "speech"

# python calculate_clus_metrics.py $CKPT_PATH_BASE "valid_true" 1 "text" "speech"

python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 0 "speech" "text"

python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 1 "speech" "text"

python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 0 "text" "text"

python calculate_clus_metrics.py $CKPT_PATH_BASE "train" 1 "text" "text"

# python calculate_clus_metrics.py $CKPT_PATH_BASE "valid_true" 0 "speech" "text"

# python calculate_clus_metrics.py $CKPT_PATH_BASE "valid_true" 1 "speech" "text"
