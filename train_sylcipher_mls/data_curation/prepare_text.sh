# text_path=/nobackup/users/junruin2/data/hubert/libri-train-clean-top-4096/feat_hubert_large_l21_ori_feats_no_sil_clsattn_bnd/top_4096
# target_dir=/nobackup/users/junruin2/data/hubert/libri-train-clean-top-4096/feat_hubert_large_l21_ori_feats_no_sil_clsattn_bnd/top_4096/km_dir_4096


text_path=/nobackup/users/junruin2/data/hubert/libri-train-clean-vad-pyphen/feat_hubert_large_l21_ori_feats_gradseg_wav2bnd/top_2048/
target_dir=/nobackup/users/junruin2/data/hubert/libri-train-clean-vad-pyphen/feat_hubert_large_l21_ori_feats_gradseg_wav2bnd/top_2048/km_dir_2048

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/train_words.txt --only-source --destdir $target_dir/text_train --srcdict $text_path/dict.txt
mv $target_dir/text_train/train.bin $target_dir/text_train/text_train.bin
mv $target_dir/text_train/train.idx $target_dir/text_train/text_train.idx

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/valid_words.txt --only-source --destdir $target_dir/text_valid --srcdict $text_path/dict.txt
mv $target_dir/text_valid/train.bin $target_dir/text_valid/text_valid.bin
mv $target_dir/text_valid/train.idx $target_dir/text_valid/text_valid.idx

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/valid_true_words.txt --only-source --destdir $target_dir/text_valid_true --srcdict $text_path/dict.txt
mv $target_dir/text_valid_true/train.bin $target_dir/text_valid_true/text_valid_true.bin
mv $target_dir/text_valid_true/train.idx $target_dir/text_valid_true/text_valid_true.idx
