import numpy as np
import joblib

topk=4096
small_size=1024
feature_type="xxx" # be consistent with the naming of the features
wrd_bnd_type="your_flag_for_vghubert_bnds" # be consistent with the naming of the word boundaries

# point to the k-means npy file trained on the 1024-word corpus
km_path_small = f'/path/to/data/libri-train-clean-top-{small_size}/feat_{feature_type}_no_sil_{wrd_bnd_type}/top_{small_size}/km_dir_{small_size}/seg_feats/km_model.npy'

# point to the k-means sklearn model trained on the 4096-word corpus, after fixing the first 1024 clusters
km_path_large = f'/path/to/data/libri-train-clean-top-{topk}/feat_{feature_type}_no_sil_${wrd_bnd_type}/top_{topk}/km_dir_{topk}/seg_feats/km_model_fckmeans_from_{small_size}'

# point to the save path of the converted k-means npy model trained on the 4096-word corpus from the sklearn model, after fixing the first 1024 clusters
save_large_path = km_path_large + '_converted.npy'

km_model_large = joblib.load(km_path_large)
C_np_large = km_model_large.cluster_centers_

C_np_small = np.load(km_path_small)

np.save(save_large_path, C_np_large)

# performs sanity checks
for i in range(small_size):
    if np.linalg.norm(C_np_large[i] - C_np_small[i]) != 0:
        print(f'ERROR on {i}-th cluster! It should be fixed!')
