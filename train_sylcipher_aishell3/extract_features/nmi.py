from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score, completeness_score 

# 0.7626793914164909 0.5355712378501435 0.6391159094747401
#feat_type = 'hubert_large_ll60k_L21'

# 0.7509425001010054 0.5288471470412404 0.6301855272619588
#feat_type = 'hubert_large_ll60k_L14'

# 0.7270848337338435 0.5140965793253512 0.611385169841329
#feat_type = 'hubert_large_ll60k_L18'

# 0.7438359438149457 0.523543843540255 0.6240438518151347
#feat_type = 'wav2vec2_zh_L9'

# 0.5708902465372606 0.4015399527722397 0.4787851737812105
#feat_type = 'xlsr2_960m_1000k_L42'

# 0.7164887735210911 0.5053270084025014 0.601715155598885
#feat_type = 'xlsr2_960m_1000k_L28'

# 0.7398377114069128 0.624700708689893 0.6774116958885164
feat_type = 'wavlm-large_L21'
x = 'test'

hyp_file = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_{feat_type}_forced_align_bnd/top_2048/km_dir_2048/seg_feats/km_idx_clsattnbndkmeans/{x}.km'
ref_file = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_{feat_type}_forced_align_bnd/top_2048/{x}_words.txt'
filelist = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_{feat_type}_forced_align_bnd/top_2048/km_dir_2048/discrete_speech_{x}_clsattnbndkmeans/file_list.txt' 
bad_filelist = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_{feat_type}_forced_align_bnd/top_2048/km_dir_2048/discrete_speech_{x}_clsattnbndkmeans/bnd_errors.txt'

with open(hyp_file, 'r') as f:
    hyps = f.read().strip().split('\n')

with open(ref_file, 'r') as f:
    refs = f.read().strip().split('\n')

with open(filelist, 'r') as f:
    filenames = f.read().strip().split('\n')
 
with open(bad_filelist, 'r') as f:
    bad_filenames = f.read().strip().split('\n')

hyp_units = []
ref_units = []
vocab = []
print(len(filenames))
for hyp, ref, fn in zip(hyps, refs, filenames):
    if not fn in bad_filenames:
        min_len = min(len(hyp.strip().split()), len(ref.split()))

        if len(hyp.strip().split()) != len(ref.split()):
            print(ref)
            print(len(hyp.strip().split()), len(ref.split()))
        hyp_units.extend(list(map(int, hyp.strip().split()))[:min_len])
        for c in ref.split()[:min_len]:
            if not c in vocab:
                vocab.append(c)
            ref_units.append(vocab.index(c))

print(len(hyp_units), len(ref_units))
assert len(hyp_units) == len(ref_units)
print(homogeneity_score(ref_units, hyp_units), completeness_score(ref_units, hyp_units), normalized_mutual_info_score(ref_units, hyp_units)) 
