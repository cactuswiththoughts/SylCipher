
start_cp=checkpoint_1103_160
all_splits=($start_cp)
# all_splits+=('checkpoint_1098_40')
# all_splits+=('checkpoint_1098_50')
# all_splits+=('checkpoint_1099_60')
# all_splits+=('checkpoint_1099_70')
# all_splits+=('checkpoint_1100_80')
# all_splits+=('checkpoint_1100_90')
# all_splits+=('checkpoint_1100_100')
# all_splits+=('checkpoint_1101_110')
# all_splits+=('checkpoint_1101_120')
# all_splits+=('checkpoint1101')
# all_splits+=('checkpoint_1102_140')
# all_splits+=('checkpoint_1102_150')
# all_splits+=('checkpoint_1103_160')


echo $all_splits

for curr_cp in $all_splits; do
    python dump_boundary.py "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats/top_1024/km_dir_1024/" --ckpt-path /nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats/top_1024/km_dir_1024/models_policy_noskip_try_debug_32/${curr_cp}.pt --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_debug_2/speecht5" --feat-name check_policy_gradient_${curr_cp} --split "speech_valid|text_valid"
done