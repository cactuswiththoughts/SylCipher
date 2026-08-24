topk=4096
tsv_folder="/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil/tsvs"
feat_dir="/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil/hubert_feats"
hubert_large_ckpt="/path/to/hubert_large_ll60k.pt"
hubert_base_ckpt="/path/to/hubert_base_ls960.pt"

bash prepare_label_no_spk_hubert.sh 1 21 $tsv_folder "$feat_dir/feat_hubert_large_L21" "$feat_dir/label_hubert_large_L21" "$feat_dir/km_model_hubert_large_L21" $hubert_large_ckpt 100

bash prepare_label_no_spk_hubert.sh 1 6 $tsv_folder "$feat_dir/feat_hubert_base_L6" "$feat_dir/label_hubert_base_L6" "$feat_dir/km_model_hubert_base_L6" $hubert_base_ckpt 100