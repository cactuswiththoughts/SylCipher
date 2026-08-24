import editdistance
import json
import logging
import numpy as np
import os
import pickle as pkl
import sys
import torch
from copy import deepcopy
from pathlib import Path
from shutil import copyfile


logging.basicConfig(
    format='%(asctime)s | %(name)s | %(message)s',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('combine_weights')


class ApplyKmeans(object):
    def __init__(self, C_np):
        self.C_np = C_np.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def combine_checkpoints(ckpt_paths, out_dir, out_ckpt_path, n_special=4):
    ckpts = [torch.load(cp) for cp in ckpt_paths]
    new_ckpt = deepcopy(ckpts[0])
    # Transfer weights for the speech prenet and postnet
    speech_keys = [
        'encoder.src_proj.weight',
        'encoder.src_proj.bias',
        'speech_encoder_prenet.encoder_prenet.0.weight',
        'speech_predictor.predictor_net.weight',
        'speech_predictor.predictor_net.bias',
        'speech_encoder_postnet.label_embs_concat',
        'speech_decoder_prenet.embed_tokens.weight',
        'speech_decoder_postnet.output_projection.weight',
    ]

    new_ckpt['model']['speech_encoder_prenet.quantizer.embedding'] = torch.from_numpy(np.load(str(out_dir / 'seg_feats' / 'km_model.npy')))
    new_ckpt['model']['speech_encoder_prenet.quantizer.embedding.data'] = torch.from_numpy(np.load(str(out_dir / 'seg_feats' / 'km_model.npy')))
    new_ckpt['cfg']['task'].data = str(out_dir)
    new_ckpt['cfg']['model'].init_clus_dir = str(out_dir / 'seg_feats' / 'km_model.npy')
    new_ckpt['cfg']['model'].init_model_dir = out_ckpt_path
    for k in speech_keys:
        p = torch.cat([ckpt['model'][k][n_special:-1] for ckpt in ckpts])
        p = torch.cat([ckpts[0]['model'][k][:n_special], p, ckpts[0]['model'][k][-1:]])
        new_ckpt['model'][k] = p
        print(k, p.shape)

    text_keys = [
        'encoder.proj.weight',
        'encoder.proj.bias',
        'text_encoder_prenet.encoder_prenet.0.weight',
        'text_predictor.predictor_net.weight',
        'text_predictor.predictor_net.bias',
        'text_encoder_postnet.label_embs_concat',
        'text_decoder_prenet.embed_tokens.weight',
        'text_decoder_postnet.output_projection.weight',
    ]

    vocab = []
    row_indices = []
    i = 0
    for src_dir in src_dirs:
        with open(src_dir / 'dict.txt', 'r') as f:
            for l in f:
                w, c = l.strip().split()
                if not w in vocab:
                    vocab.append(w)
                    row_indices.append(i)
                i += 1

    for k in text_keys:
        p = torch.cat([ckpt['model'][k][n_special:] for ckpt in ckpts])
        p = p[row_indices]
        p = torch.cat([ckpts[0]['model'][k][:n_special], p, ckpts[0]['model'][k][-2:]])
        new_ckpt['model'][k] = p
        print(k, p.shape)
    
    Path(out_ckpt_path).parent.mkdir(parents=True, exist_ok=True)   
    torch.save(new_ckpt, out_ckpt_path)


def combine_dict(src_dirs, lang_ids, out_dir):
    km_size = np.load(out_dir / 'seg_feats' / 'km_model.npy').shape[0]
    with open(out_dir / 'dict.audio.txt', 'w') as f_d1,\
        open(out_dir / 'dict.audio0.txt', 'w') as f_d2:
        for lang_id in lang_ids:
            print(f'{lang_id} 0', file=f_d1)
            print(f'{lang_id} 0', file=f_d2)

        for i in range(km_size):
            print(f'{i} 1', file=f_d1)
            print(f'{i} 1', file=f_d2)

    offset = 0
    with open(out_dir / 'dict.txt', 'w') as f_d:
        vocab = []
        for lang_id in lang_ids: 
            print(f'{lang_id} 0', file=f_d)

        for src_dir, lang_id in zip(src_dirs, lang_ids):
            with open(src_dir / 'dict.txt', 'r') as f:
                for l in f:
                    w, c = l.strip().split()
                    if w != '<OOV>':
                        print(' '.join([f'{lang_id}_{w}', c]), file=f_d)
                        vocab.append(w)

        if len(vocab) < km_size:
            gap = km_size - len(vocab)
            for i in range(gap):
                print(f'<placeholder{i}> 0', file=f_d)
        print('<OOV> 0', file=f_d)
        print(f'Combined vocab size: {len(vocab)}')
        logger.info(f'Combined vocab size: {len(vocab)}')


def reassign_cluster(src_dirs, out_dir):
    # Concat centroids
    centroids = [np.load(str(src_dir / 'seg_feats' / 'km_model.npy')) for src_dir in src_dirs]
    print([c.shape for c in centroids])  # XXX
    clus_offset = 0
    clus_offsets = []
    for c in centroids:
        clus_offsets.append(c.shape[0])
        clus_offset += c.shape[0]

    C = np.concatenate(centroids)
    logger.info(f'centroid size: {C.shape}')
    (out_dir / 'seg_feats').mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / 'seg_feats' / 'km_model.npy'), C)
    Path(out_dir / 'discrete_speech_train_clsattnbndkmeans').mkdir(exist_ok=True, parents=True)
    for i in range(len(src_dirs)):
        Path(out_dir / f'discrete_speech_train{i}_clsattnbndkmeans').mkdir(exist_ok=True, parents=True)
        Path(out_dir / f'discrete_speech_test{i}_clsattnbndkmeans').mkdir(exist_ok=True, parents=True)

    vocab = []
    with open(out_dir / 'discrete_speech_train_clsattnbndkmeans' / 'data.km', 'w') as f_train:
        for i, (src_dir, clus_offset) in enumerate(zip(src_dirs, clus_offsets)):
            # Train
            with open(src_dir / 'discrete_speech_train_clsattnbndkmeans' / 'data.km', 'r') as f_in,\
                open(out_dir / f'discrete_speech_train{i}_clsattnbndkmeans' / 'data.km', 'w') as f_train_mono:
                lines = f_in.readlines()
                for l in lines:
                    labels = [int(lbl)+clus_offset for lbl in l.strip().split()]
                    labels = list(map(str, labels))
                    print(' '.join(labels), file=f_train)
                    print(' '.join(labels), file=f_train_mono)

            # Test
            (out_dir / f'discrete_speech_test{i}_clsattnbndkmeans').mkdir(exist_ok=True, parents=True)
            with open(src_dir / 'discrete_speech_test_clsattnbndkmeans' / 'data.km', 'r') as f_in,\
                open(out_dir / f'discrete_speech_test{i}_clsattnbndkmeans' / 'data.km', 'w') as f_test:
                lines = f_in.readlines()
                for l in lines:
                    labels = [int(lbl)+clus_offset for lbl in l.strip().split()]
                    labels = list(map(str, labels))
                    print(' '.join(labels), file=f_test)


def reassign_frame_cluster(src_dirs, out_dir):
    out_train_dir = out_dir / 'discrete_speech_train_clsattnbndkmeans'
    km_offset = 0
    with open(out_train_dir / 'data_frame_clus.km', 'w') as f_tr:
        for i, src_dir in enumerate(src_dirs):
            train_dir = src_dir / 'discrete_speech_train_clsattnbndkmeans'
            feat_name = '_'.join(src_dir.parent.parent.name.split('_')[:3])
            C = np.load(str(src_dir / '../../..' / feat_name / 'km_model.npy'))
            km_size = C.shape[0]

            with open(train_dir / 'data_frame_clus.km', 'r') as f_tr_in,\
                open(out_train_dir / f'../discrete_speech_train{i}_clsattnbndkmeans' / 'data_frame_clus.km', 'w') as f_tr_mono:
                lines = f_tr_in.read().strip().split('\n')
                for l in lines:
                    labels = ' '.join([str(int(y)+km_offset) for y in l.split()])
                    print(labels, file=f_tr)
                    print(labels, file=f_tr_mono)

            test_dir = src_dir / 'discrete_speech_test_clsattnbndkmeans'
            out_test_dir = out_dir / f'discrete_speech_test{i}_clsattnbndkmeans'
            with open(test_dir / 'data_frame_clus.km', 'r') as f_te_in,\
                open(out_test_dir / 'data_frame_clus.km', 'w') as f_te:
                lines = f_te_in.read().strip().split('\n')
                for l in lines:
                    labels = ' '.join([str(int(y)+km_offset) for y in l.split()])
                    print(labels, file=f_te)
            km_offset += km_size
    print(f'Total number of frame clusters: {km_offset}')


def combine_metadata(src_dirs, lang_ids, out_dir):
    bnd_errors = []
    bound_dict = {}
    meta_dict = {}
    split = 'discrete_speech_train_clsattnbndkmeans'
    with open(out_dir / split / 'file_list.txt', 'w') as f_list,\
        open(out_dir / split / 'bnd_errors.txt', 'w') as f_bnd_err,\
        open(out_dir / split / 'data_dict_bnd.pkl', 'wb') as f_bnd,\
        open(out_dir / split / 'meta_data.json', 'w') as f_meta,\
        open(out_dir / split / 'data.lengths', 'w') as f_len,\
        open(out_dir / split / 'lang_ids.txt', 'w') as f_lang:
        for i, (src_dir, lang_id) in enumerate(zip(src_dirs, lang_ids)):
            out_split = f'discrete_speech_train{i}_clsattnbndkmeans'
            with open(src_dir / split / 'file_list.txt', 'r') as f,\
                open(out_dir / out_split / 'lang_ids.txt', 'w') as f_lang_mono: 
                lines = f.read().strip().split('\n')
                print('\n'.join(lines), file=f_list)
                print('\n'.join([lang_id]*len(lines)), file=f_lang)
                copyfile(src_dir / split / 'file_list.txt', out_dir / out_split / 'file_list.txt')
                print('\n'.join([lang_id]*len(lines)), file=f_lang_mono)

            with open(src_dir / split / 'bnd_errors.txt', 'r') as f:
                if len(f.read().strip()):
                    print(f.read().strip(), file=f_bnd_err)
                copyfile(src_dir / split / 'bnd_errors.txt', out_dir / out_split / 'bnd_errors.txt')

            with open(src_dir / split / 'data_dict_bnd.pkl', 'rb') as f:
                cur_bound_dict = pkl.load(f)
                bound_dict.update(cur_bound_dict)
                copyfile(src_dir / split / 'data_dict_bnd.pkl', out_dir / out_split / 'data_dict_bnd.pkl')

            with open(src_dir / split / 'meta_data.json', 'r') as f,\
                open(out_dir / out_split / 'meta_data.json', 'w') as f_mono:
                cur_meta_dict = json.load(f)
                for fn in cur_meta_dict:
                    meta_dict[fn] = []
                    for start, end, token in cur_meta_dict[fn]:
                        if token == '<OOV>':
                            meta_dict[fn].append([start, end, '<OOV>'])
                        else:
                            meta_dict[fn].append([start, end, f'{lang_id}_{token}'])
                json.dump(meta_dict, f_mono, indent=2)

            with open(src_dir / split / 'data.lengths', 'r') as f:
                lines = f.read().strip()
                print(lines, file=f_len)
                copyfile(str(src_dir / split / 'data.lengths'), str(out_dir / out_split / 'data.lengths'))

        pkl.dump(bound_dict, f_bnd)
        json.dump(meta_dict, f_meta, indent=2)
    
    for i, (src_dir, lang_id) in enumerate(zip(src_dirs, lang_ids)):
        split = 'discrete_speech_test_clsattnbndkmeans'
        out_split = f'discrete_speech_test{i}_clsattnbndkmeans'
        copyfile(str(src_dir / split / 'file_list.txt'), str(out_dir / out_split / 'file_list.txt'))
        copyfile(str(src_dir / split / 'bnd_errors.txt'), str(out_dir / out_split / 'bnd_errors.txt'))
        copyfile(str(src_dir / split / 'data_dict_bnd.pkl'), str(out_dir / out_split / 'data_dict_bnd.pkl'))
        copyfile(str(src_dir / split / 'data.lengths'), str(out_dir / out_split / 'data.lengths'))
        meta_dict = {}
        with open(src_dir / split / 'meta_data.json', 'r') as f_in,\
            (out_dir / out_split / 'meta_data.json').open('w', encoding='utf-8') as f_out:
            cur_meta_dict = json.load(f_in)
            for fn in cur_meta_dict:
                meta_dict[fn] = []
                for start, end, token in cur_meta_dict[fn]:
                    if token == '<OOV>':
                        meta_dict[fn].append([start, end, '<OOV>'])
                    else:
                        meta_dict[fn].append([start, end, f'{lang_id}_{token}'])
            json.dump(meta_dict, f_out, indent=4, ensure_ascii=False)

        with open(src_dir / split / 'file_list.txt', 'r') as f,\
            open(out_dir / out_split / 'lang_ids.txt', 'w') as f_lang:
            lines = f.read().strip().split('\n')
            print('\n'.join([lang_id]*len(lines)), file=f_lang)
            

def combine_aux_targets(src_dirs, lang_ids, out_dir):
    split = 'discrete_speech_train_clsattnbndkmeans'
    with open(out_dir / split / 'data.txt', 'w') as f_tgt:
        for i, (src_dir, lang_id) in enumerate(zip(src_dirs, lang_ids)):
            out_split = f'discrete_speech_train{i}_clsattnbndkmeans'
            with open(src_dir / split / 'data.txt', 'r') as f,\
                open(out_dir / out_split / 'data.txt', 'w') as f_mono:
                lines = f.read().strip().split('\n')
                for l in lines:
                    tokens = l.strip().split('\t')[-1].split()
                    new_tokens = [f'{lang_id}_{token}' if token != '<OOV>' else '<OOV>' for token in tokens]
                    print(' '.join([lang_id]+new_tokens), file=f_tgt)
                    print(' '.join([lang_id]+new_tokens), file=f_mono)


def combine_text_data(src_dirs, lang_ids, out_dir):
    split = 'train'
    with open(out_dir / f'../{split}_words.txt', 'w') as f_tr:
        for i, (src_dir, lang_id) in enumerate(zip(src_dirs, lang_ids)):
            out_split = f'train{i}'
            print(src_dir / f'../{split}_words.txt')
            with open(src_dir / f'../{split}_words.txt', 'r') as f,\
                open(out_dir / f'../{out_split}_words.txt', 'w' ) as f_tr_mono:
                lines = f.read().strip().split('\n')
                for l in lines:
                    new_tokens = [f'{lang_id}_{token}' if token != '<OOV>' else '<OOV>' for token in l.strip().split()]
                    print(' '.join([lang_id]+new_tokens), file=f_tr)
                    print(' '.join([lang_id]+new_tokens), file=f_tr_mono)

    split = 'test'
    for i, (src_dir, lang_id) in enumerate(zip(src_dirs, lang_ids)):
        with open(src_dir / f'../{split}_words.txt', 'r') as f,\
            open(out_dir / f'../{split}{i}_words.txt', 'w') as f_te:
            lines = f.read().strip().split('\n')
            for l in lines:
                new_tokens = [f'{lang_id}_{token}' if token != '<OOV>' else '<OOV>' for token in l.strip().split()]
                print(' '.join([lang_id]+new_tokens), file=f_te)


lang_ids = ['<cmn>', '<yue>', '<nan>']
src_dirs = [
    Path('/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048'),
    Path('/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mdcc_vad/feat_xeus_L18_sylber_bnd/top_608/km_dir_608'),
    Path('/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/suisiann_vad/feat_xeus_L18_sylber_ft_bnd/top_1024/km_dir_1024'),
]

out_dir = Path('/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048')

logger.info("Re-assign cluster units")
#reassign_cluster(src_dirs, out_dir)
#reassign_frame_cluster(src_dirs, out_dir)

logger.info("Create new dictionaries")
#combine_dict(src_dirs, lang_ids, out_dir)

logger.info("Concat filelist, bnd_errors, boundaries, etc.")
#combine_metadata(src_dirs, lang_ids, out_dir)

logger.info("Combine text data")
combine_text_data(src_dirs, lang_ids, out_dir)

logger.info("Combine auxiliary labels")
combine_aux_targets(src_dirs, lang_ids, out_dir)
