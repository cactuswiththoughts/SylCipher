import numpy as np
import torch
import torchaudio
import json
import os


def get_data_buckeye(file_tsv, file_boundary_json, parent_dir=None, load_wav=True):

    with open(file_tsv, 'r') as f:
        tsv_lines = f.readlines()
        if parent_dir is None: 
            parent_dir = tsv_lines[0].strip()
            file_lines = tsv_lines[1:]
        else:
            print(f'No header, parent_dir: {parent_dir}')
            file_lines = tsv_lines

    with open(file_boundary_json, 'r') as f:
        boundary_dict = json.load(f)

    wavs = [os.path.join(parent_dir, file_line.strip().split()[0]) for file_line in file_lines]
    wavs_fns = [file_line.strip().split()[0] for file_line in file_lines]
    all_wavs = []
    all_bounds = []
    all_fns = []
    # rp = np.random.permutation(len(wavs))
    # wavs = [wavs[i] for i in rp]
    # wavs_fns = [wavs_fns[i] for i in rp]
    for idx, wav in enumerate(wavs):
        if load_wav:
            waveform, sample_rate = torchaudio.load(wav)
        else:
            waveform = None

        if wavs_fns[idx] in boundary_dict:
            words = boundary_dict[wavs_fns[idx]]
            bounds = [(int(w[0]), int(w[1])) for w in words]
        else:
            print(f'gt boundary for {wavs_fns[idx]} not found')
            bounds = [(0, 16000)]

        # if len(bounds) > 0:
        all_wavs.append(waveform)
        all_bounds.append(bounds)
        all_fns.append(wavs_fns[idx])
    return all_wavs, all_bounds, all_fns


def get_emb(wavs, model, layer=-1, feat_idx=-1):
    es = []
    for waveform in wavs:
        e = embed(waveform, model, layer, feat_idx)
        es.append(e)
    return es

def get_single_emb(wavform, model, layer=-1, feat_idx=-1):
    return embed(wavform, model, layer, feat_idx)

def embed(y, model, extract_layer=-1, feat_idx = -1):
    with torch.no_grad():
        model.eval()
        y = torch.Tensor(y).cuda()
        x, _ = model.extract_features(y)
        x = x[extract_layer]
        if not feat_idx == -1:
            x = x[:,:,feat_idx]
    return x.data.cpu().numpy()[0]


def get_model(arc):
    if arc == 'BASE': #12 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        my_dim = 768
    elif arc == 'LARGE': #24 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        my_dim = 1024
    elif arc == 'LARGE_LV60K': #24 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        my_dim = 1024
    elif arc == 'XLSR53': #24 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        my_dim = 1024
    elif arc == 'HUBERT_BASE':
        bundle = torchaudio.pipelines.HUBERT_BASE
        my_dim = 768
    elif arc == 'HUBERT_LARGE':
        bundle = torchaudio.pipelines.HUBERT_LARGE
        my_dim = 1024
    elif arc == 'HUBERT_XLARGE':
        bundle = torchaudio.pipelines.HUBERT_XLARGE
        my_dim = 1280 #?
    else:   
        bundle = torchaudio.pipelines.WAV2VEC2_BASE #WAV2VEC2_BASE WAV2VEC2_LARGE WAV2VEC2_LARGE_LV60K WAV2VEC2_XLSR53
        my_dim = 768
    return bundle.get_model(), my_dim


def get_bounds(boundaries):
    l = [0]
    for i in range(len(boundaries)-1):
        l.append((boundaries[i][1] + boundaries[i+1][0])//2)
    l.append(boundaries[-1][1])
    return l
