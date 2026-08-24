#!/bin/bash
#SBATCH -J run_dump_embedding
#SBATCH -o run_dump_embedding_%j.out
#SBATCH -e run_dump_embedding_%j.err
#SBATCH --mail-user=junruin2@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20

# DATA_DIR=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/models_diff_bnd_bnd_gradient_DEBUGGING_from_scratch_debug_requires_grad_all_true_noaddenc_pass_params_no_use_softmax_codebook_0.3/checkpoint_best_49.283.pt
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/models_diff_bnd_bnd_gradient_DEBUGGING_from_scratch_debug_requires_grad_all_true_noaddenc_pass_params_no_use_softmax_codebook_0.3_nobndgrad/checkpoint_best_45.213.pt
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/models_diff_bnd_bnd_gradient_DEBUGGING_from_scratch_debug_requires_grad_all_true_noaddenc_pass_params_no_use_softmax_codebook_0.3_retry_with_pretrained_quantizer_code_loss_1/checkpoint_best.pt
# DATA_DIR=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_clsattn_bnd/top_1024/km_dir_1024/
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_clsattn_bnd/top_1024/km_dir_1024/models_codebook_0.3_pretrain_redo_boundary_classifier_as_cnn/checkpoint_best_45.883.pt
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5

# DATA_DIR=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/models_codebook_0.3_gradsegkmeans_reset_lr_1/checkpoint_best_61.346.pt
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5
# DATA_DIR=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_gradseg/top_1024/km_dir_1024/
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_gradseg/top_1024/km_dir_1024/models_codebook_0.3_vghubertkmeans/checkpoint_best.pt
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5


# DATA_DIR="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/"
# # CKPT_PATH="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans/checkpoint_last_50.901.pt"
# CKPT_PATH="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans/checkpoint_last.pt"
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5


# DATA_DIR="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/"
# # CKPT_PATH="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_diff_bnd/checkpoint_best_55.255.pt"
# CKPT_PATH="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_diff_bnd_retry/checkpoint_best.pt"
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5

# DATA_DIR="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_w2b_diff_bnd/top_1024/km_dir_1024/"
# CKPT_PATH="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_w2b_diff_bnd/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_no_load_quantizer_retry_1/checkpoint_last_60.875.pt"
# CKPT_PATH="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_w2b_diff_bnd/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_no_load_quantizer_retry_1/checkpoint_best.pt"
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_for_eval_w2b_diff_bnd/checkpoint_last_50.901.pt
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_gradseg/top_1024/km_dir_1024/models_codebook_0.3_vghubertkmeans_for_eval_w2b_diff_bnd/checkpoint_best.pt
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_diff_bnd_retry_for_eval_w2b_diff_bnd/checkpoint_best.pt
# CKPT_PATH=/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_w2b_diff_bnd/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_no_load_quantizer_retry_1_diff_bnd/checkpoint_best.pt

# DATA_DIR="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/"
# CKPT_PATH="/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_no_load_quantizer/checkpoint_last.pt"
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5


# DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/"
# CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gt_bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans/checkpoint_best.pt"


# DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/"
# # CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans/checkpoint_best_49.832.pt"
# # CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans_diff_bnd_1/checkpoint_best_62.126.pt"
# # CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans/checkpoint_best.pt"

# # DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_clsattn_bnd/top_1024/km_dir_1024/"
# # CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_clsattn_bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans/checkpoint_best.pt"
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5

# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true|text_valid_true" --postfix "valid_true" --tgt-enc-layer 0

# echo "dumping TRAIN"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train|text_train" --postfix "train" --tgt-enc-layer 0

# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true|text_valid_true" --postfix "valid_true" --tgt-enc-layer 1

# echo "dumping TRAIN"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train|text_train" --postfix "train" --tgt-enc-layer 1

# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true|text_valid_true" --postfix "valid_true" --tgt-enc-layer 2

# echo "dumping TRAIN"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train|text_train" --postfix "train" --tgt-enc-layer 2

# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true|text_valid_true" --postfix "valid_true" --tgt-enc-layer 3

# echo "dumping TRAIN"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train|text_train" --postfix "train" --tgt-enc-layer 3

# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true|text_valid_true" --postfix "valid_true" --tgt-enc-layer 4

# echo "dumping TRAIN"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train|text_train" --postfix "train" --tgt-enc-layer 4

# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true|text_valid_true" --postfix "valid_true" --tgt-enc-layer 5

# echo "dumping TRAIN"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train|text_train" --postfix "train" --tgt-enc-layer 5

# DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gradseg/top_1024/km_dir_1024/"
# CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_gradseg/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans/checkpoint_best.pt"
# DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_w2b_diff_bnd/top_1024/km_dir_1024/"
# CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_w2b_diff_bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans/checkpoint_last_66.762.pt"
# USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5


# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true" --postfix "valid_true" --use-discrete-labels 1

# # echo "dumping VALID"
# # python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_clsattnbndkmeans|text_valid" --postfix "valid" --use-discrete-labels 1

# echo "dumping TRAIN"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train_clsattnbndkmeans|text_train" --postfix "train" --use-discrete-labels 1

# cd ../
# python word_translation_code2word.py


DATA_DIR="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_clsattn_bnd/top_1024/km_dir_1024/"
CKPT_PATH="/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_clsattn_bnd/top_1024/km_dir_1024/model_codebook_0.3_orikmeans_train_other_500_unpaired/checkpoint_last.pt"
USER_DIR=/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5
UNPAIRED_TEXT_PATH="/nobackup/users/junruin2/fairseq_expspring2022/examples/wav2vec/unsupervised/librispeech_train_other_500_lm/lm.upper.top1024.lid.txt"


# echo "dumping VALID_TRUE"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true|text_valid_true" --postfix "valid_true" --use-discrete-labels 1

# echo "dumping VALID"
# python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_clsattnbndkmeans|text_valid" --postfix "valid" --use-discrete-labels 1

echo "dumping TRAIN"
python dump_embedding.py $DATA_DIR --ckpt-path $CKPT_PATH --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_train|text_train_unpaired_train_other_500" --postfix "train" --use-unpaired-text 1 --unpaired-text-path $UNPAIRED_TEXT_PATH --use-discrete-labels 1

cd ../
python word_translation_code2word.py