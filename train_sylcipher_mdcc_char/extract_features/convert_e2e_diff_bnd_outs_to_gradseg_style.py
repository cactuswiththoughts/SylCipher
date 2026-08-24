import os
import pickle
import numpy as np
import copy


topk=2048
feature_type='xxx' # be consistent with the naming of the features
wrd_bnd_type='your_flag_for_bnd_used' # be consistent with the naming of the word
# point to your data directory for a specific boundary type
datadir=f'/path/to/data/libri-train-clean-top-{topk}/feat_{feature_type}_no_sil_{wrd_bnd_type}/top_{topk}/km_dir_{topk}'
audio_dir=f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/'
for which_set in ['train', 'valid', 'valid_true']:
    # point to the outputs of extract_e2e_bnds.sh
    predicted_boundary_path = f'{datadir}/model_codebook_0.0_clsattn_bnd_kmeans_encoder_only_retry_2024_2l_with_diff_bnd/predicted_boundaries_{which_set}.pkl'

    # point to where to save the output boundary files
    new_manifest_path = f'{audio_dir}/your_saved_dir_for_jstti_e2e_bnds/'
    
    os.makedirs(new_manifest_path, exist_ok = True)
    # point to the original boundaries before JSTTI E2E-refinement, in case you need to replace the boundary predictions when JSTTI E2E-refinement fails on some utterances
    manifest_path = f'{audio_dir}/your_flag_for_gradseg_wav2bnd_bnds/'

    # files to exclude
    bad_fns_txt = f'/path/to/{which_set}_bnd_errors.txt'


    predicted_boundaries = []

    with open(predicted_boundary_path, 'rb') as fb:
        predicted_boundaries_dict = pickle.load(fb)
    filenames = list(predicted_boundaries_dict.keys())

    bad_fns = []
    with open(bad_fns_txt, 'r') as f:
        bad_lines = f.readlines()

    for bad_line in bad_lines:
        bad_fns.append(bad_line.strip().split()[0])

    bad_fns = set(bad_fns)
    print(bad_fns)

    bad_idx =[]
    print(len(filenames))
    for idx, fn in enumerate(filenames):
        # curr_prediction = get_peak_preds(torch.nn.Sigmoid()(predicted_boundaries_dict[fn]).cpu().numpy(),0.4,0.5,1)
        curr_prediction = predicted_boundaries_dict[fn]    
        # if idx < 10:
        #     print(curr_prediction)
        #     print(predicted_boundaries_dict[fn])
        if curr_prediction.dim() != 0 and len(curr_prediction) > 1 and fn not in bad_fns:
            curr_prediction[0] = 1
            curr_prediction[-1] = 1
            predicted_boundaries.append(curr_prediction.numpy().astype('int32'))

        else:
            if curr_prediction.dim() == 0:
                print('huh', filenames[idx])
            bad_idx.append(idx)
            print('bad!', fn)

    new_save_boundary_dict = {}
    with open(os.path.join(manifest_path, f'{which_set}_utts.pkl'), 'rb') as fb:
        boundary_dict = pickle.load(fb)

    for idx, filename in enumerate(filenames):
        if idx not in bad_idx:
            numpy_bnd = predicted_boundaries_dict[filenames[idx]].numpy()
            numpy_bnd[0] = 1
            new_boundary_locs = np.where(numpy_bnd == 1)[0] * 2


            boundaries = copy.deepcopy(boundary_dict[filename])

            boundaries['seg_bound'] = np.zeros(len(boundaries['seg_bound']))
            boundaries['seg_bound'][new_boundary_locs] = 1

            new_save_boundary_dict[filenames[idx]] = boundaries

        else:
            # print(filename)
            boundaries = copy.deepcopy(boundary_dict[filename])

            new_save_boundary_dict[filenames[idx]] = boundaries


    with open(os.path.join(new_manifest_path, f'{which_set}_utts.pkl'), 'wb') as fwb:
        pickle.dump(new_save_boundary_dict, fwb)