# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pickle
from pathlib import Path
from tqdm import tqdm
import data_loader
import eval_segmentation
import fairseq
from syllablelm.data2vec.data.modality import Modality
from syllablelm.data2vec.models.data2vec2 import Data2VecMultiModel
from syllablelm.hubert.models.hubert import HubertModel
from types import SimpleNamespace

THRESHOLD = 1 / .10 / 50.
FULL_MODELS_DICT = {
    '8.33Hz': {
        'delta': 0.0033,
        'quantile': 0.75,
    },
    '6.25Hz': {
        'delta': 0.0028,
        'quantile': 0.75,
    },
    '5.0Hz': {
        'delta': 0.0019,
        'quantile': 0.75,
    },
}

d2v2_config = SimpleNamespace(
    **{'_name': 'data2vec_multi', 'loss_beta': 0.0, 'loss_scale': None, 'depth': 8, 'start_drop_path_rate': 0.0, 'end_drop_path_rate': 0.0, 'num_heads': 12,
       'norm_eps': 1e-05, 'norm_affine': True, 'encoder_dropout': 0.1, 'post_mlp_drop': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0,
       'dropout_input': 0.0, 'layerdrop': 0.05, 'embed_dim': 768, 'mlp_ratio': 4.0, 'layer_norm_first': False, 'average_top_k_layers': 8,
       'end_of_block_targets': False, 'clone_batch': 8, 'layer_norm_target_layer': False, 'batch_norm_target_layer': False, 'instance_norm_target_layer': True,
       'instance_norm_targets': False, 'layer_norm_targets': False, 'ema_decay': 0.999, 'ema_same_dtype': True, 'log_norms': True, 'ema_end_decay': 0.99999,
       'ema_anneal_end_step': 75000, 'ema_encoder_only': False, 'max_update': 400000, 'modalities': SimpleNamespace(**{'_name': None, 'audio': SimpleNamespace(
            **{'type': Modality.AUDIO, 'prenet_depth': 4, 'prenet_layerdrop': 0.05, 'prenet_dropout': 0.1, 'start_drop_path_rate': 0.0,
               'end_drop_path_rate': 0.0, 'num_extra_tokens': 0, 'init_extra_token_zero': True, 'mask_noise_std': 0.01, 'mask_prob_min': None, 'mask_prob': 0.5,
               'inverse_mask': False, 'mask_prob_adjust': 0.05, 'keep_masked_pct': 0.0, 'mask_length': 5, 'add_masks': False, 'remove_masks': False,
               'mask_dropout': 0.0, 'encoder_zero_mask': True, 'mask_channel_prob': 0.0, 'mask_channel_length': 64, 'ema_local_encoder': False,
               'local_grad_mult': 1.0, 'use_alibi_encoder': True, 'alibi_scale': 1.0, 'learned_alibi': False, 'alibi_max_pos': None,
               'learned_alibi_scale': True, 'learned_alibi_scale_per_head': True, 'learned_alibi_scale_per_layer': False, 'num_alibi_heads': 12,
               'model_depth': 8, 'decoder': SimpleNamespace(
                    **{'decoder_dim': 384, 'decoder_groups': 16, 'decoder_kernel': 7, 'decoder_layers': 4, 'input_dropout': 0.1, 'add_positions_masked': False,
                       'add_positions_all': False, 'decoder_residual': True, 'projection_layers': 1, 'projection_ratio': 2.0,
                       'channel_mult': [1, 0.5, 0.25, 0.25, 0.25], 'decoder_transformer_layers': 4}), 'extractor_mode': 'layer_norm',
               'feature_encoder_spec': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_pos_width': 95, 'conv_pos_groups': 16,
               'conv_pos_depth': 5, 'conv_pos_pre_ln': False})}), 'shared_decoder': None, 'min_target_var': 0.1, 'min_pred_var': 0.01,
       'supported_modality': Modality.AUDIO, 'mae_init': False, 'seed': 1, 'skip_ema': False, 'cls_loss': 0.0, 'recon_loss': 0.0, 'd2v_loss': 1.0,
       'decoder_group': False}
)

hubert_base_ls960_config = SimpleNamespace(activation_dropout=0.0, activation_fn='gelu', adam_betas='(0.9,0.98)', adam_eps=1e-06, all_gather_list_size=16384, arch='hubert', attention_dropout=0.1, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_suffix='', clip_norm=10.0, conv_bias=False, conv_feature_layers='[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', conv_pos=128, conv_pos_groups=16, cpu=False, criterion='hubert', curriculum=0, data='/checkpoint/wnhsu/data/librispeech/960h/iter/250K_50hz_km100_mp0_65_v2', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method='tcp://learnfair1408:55498', distributed_no_spawn=True, distributed_port=55498, distributed_rank=0, distributed_world_size=32, distributed_wrapper='DDP', dropout=0.1, dropout_features=0.1, dropout_input=0.1, empty_cache_freq=0, enable_padding=False, encoder_attention_heads=12, encoder_embed_dim=768, encoder_ffn_embed_dim=3072, encoder_layerdrop=0.05, encoder_layers=12, end_learning_rate=0.0, extractor_mode='default', fast_stat_sync=False, feature_grad_mult=0.1, final_dim=256, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=True, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, keep_best_checkpoints=-1, keep_interval_updates=1, keep_last_epochs=-1, label_rate=50.0, labels="['layer6.km500']", layer_norm_first=False, localsgd_frequency=3, log_format='json', log_interval=1, log_keys='["prob_perplexity", "code_perplexity", "temp"]', logit_temp=0.1, loss_weights='[10,]', lr=[0.0005], lr_scheduler='polynomial_decay', mask_channel_length=10, mask_channel_min_space=1, mask_channel_other=0, mask_channel_prob=0, mask_channel_selection='static', mask_length=10, mask_min_space=1, mask_other=0.0, mask_prob=0.8, mask_selection='static', max_epoch=0, max_sample_size=250000, max_sentences=None, max_sentences_valid=None, max_tokens=1400000, max_tokens_valid=1400000, max_update=400000, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1, min_sample_size=32000, model_parallel_size=1, no_epoch_checkpoints=True, no_last_checkpoints=False, no_mask_channel_overlap=False, no_mask_overlap=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_seed_provided=False, normalize=False, nprocs_per_node=8, num_workers=6, optimizer='adam', optimizer_overrides='{}', patience=-1, power=1.0, pred_masked_weight=1.0, pred_nomask_weight=0.0, profile=False, quantization_config_path=None, required_batch_size_multiple=8, reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', sample_rate=16000, save_dir='/checkpoint/wnhsu/w2v/hubert_iter/250K_50hz_km100_mp0_65_v2/4n_50hz.MU400k.no_c10d.layer6_km500.s1337.mxsz250000.lr0_0005.wstep32000.ufreq1.maxtok1400000.fgm0_1.gnorm10_0.pmw1_0.puw0_0.lw10_.fd256.ml10.mp0_8.mo0.drpl0_05.drpi0_1.drpf0_1.ngpu32', save_interval=1, save_interval_updates=25000, scoring='bleu', seed=1337, sentence_avg=False, skip_invalid_size_inputs_valid_test=True, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, target_glu=False, task='hubert_pretraining', tensorboard_logdir='', threshold_loss_scale=None, tokenizer=None, total_num_update=400000, tpu=False, train_subset='train', update_freq=[1], use_bmuf=False, use_old_adam=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=5, validate_interval_updates=10000, warmup_updates=32000, weight_decay=0.01, skip_masked=False, skip_nomask=True, required_seq_len_multiple=2, layer_type='transformer', checkpoint_activations=False, untie_final_proj=True)

hubert_large_ll60k_config = SimpleNamespace(**{'_name': 'hubert', 'label_rate': 50, 'extractor_mode': 'layer_norm', 'encoder_layers': 24, 'encoder_embed_dim': 1024, 'encoder_ffn_embed_dim': 4096, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'dropout_features': 0.0, 'final_dim': 768, 'untie_final_proj': True, 'layer_norm_first': True, 'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', 'conv_bias': False, 'logit_temp': 0.1, 'target_glu': False, 'feature_grad_mult': 1.0, 'mask_length': 10, 'mask_prob': 0.8, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995], 'skip_masked': False, 'skip_nomask': True, 'required_seq_len_multiple': 2, 'layer_type': 'transformer', 'checkpoint_activations': False})
  

def cossim(x,y):
    return (x*y).sum(-1)/(((x**2).sum(-1)+1e-8)**.5)/(((y**2).sum(-1)+1e-8)**.5)


class SylBoostFeatureReader:
    def __init__(
            self,
            sylboost_checkpoint,
            kmeans_centroids_path,
            agglom_indices_path,
            model_key,
            delta=None,
            quantile=None,
            merge_threshold=0.8,
            norm_threshold=2.6,
    ):
        state_dict = torch.load(sylboost_checkpoint)

        self.is_d2v2 = False
        if 'hubert' in sylboost_checkpoint: 
            hubert_model = HubertModel(hubert_large_ll60k_config if 'hubert_large_ll60k' in sylboost_checkpoint else hubert_base_ls960_config)
            hubert_model = hubert_model.eval().cuda().half()
            hubert_model.load_state_dict({k[len('model.'):]: v for k, v in state_dict['model_seg'].items()})
            self.model = hubert_model
            self.encoder_layers = 24 if 'hubert_large_ll60k' in sylboost_checkpoint else 12
        else:
            self.is_d2v2 = True
            d2v2_model = Data2VecMultiModel(d2v2_config, [Modality.AUDIO])
            d2v2_model = d2v2_model.cuda().eval().half()
            d2v2_model.load_state_dict({k[len('model.'):]: v for k, v in state_dict['model_seg'].items() if k.startswith('model.')})

            self.model = d2v2_model
            self.encoder_layers = 12
        print(f'self.encoder_layers: {self.encoder_layers}')

        self.norm_threshold = norm_threshold
        self.merge_threshold = merge_threshold

        self.model_key = model_key

    @torch.no_grad()
    def forward(
            self,
            x,
    ):
        # Input:
        # x : (b, t) batched waveform tensor at 16000Hz.
        # Returns:
        # features: (b, n, d) raw data2vec2 features
        # segments: list of length b, each item is boundaries with two columns
        #   # 0th col: start boundary idx (inclusive). 1st col: end boundary idx (exclusive).
        if self.is_d2v2:
            features = self.model(x.half(), mode=None, mask=False, features_only=True, remove_extra_tokens=True, out_layer=-2)['x']
        else:
            features = self.model.forward(x.half(), features_only=True, mask=False, output_layer=self.encoder_layers-2)['x'].float()

        segments = []
        for feat in features:
            segment = self.get_segment(feat.cpu().numpy(), self.norm_threshold, self.merge_threshold)
            segments.append(segment)

        result = {
            'features': features,
            'segments': segments,
        }
        
        return result

    def get_segment(self, states, normthreshold, mergethreshold):
        states = (states - states.mean(axis=0, keepdims=True)) / states.std(axis=0, keepdims=True)

        # states: (L, d)
        norms = ((states**2).sum(-1)+1e-8)**.5

        mask = norms>=normthreshold

        curr = 0
        seg_cnt = 0
        segments = []
        s=-1
        midboundaries = []
        for i in range(len(states)):
            if ~mask[i]:
                seg_cnt = 0
                if s>-1:
                    segments.append([s,i])
                s = -1
                curr = 0
                seg_cnt = 0
            else:
                if seg_cnt == 0:
                    curr = states[i]
                    seg_cnt+=1
                    s = i
                else:
                    sim = cossim(curr, states[i])
                    if sim >= mergethreshold:
                        curr = (curr*seg_cnt+states[i])/(seg_cnt+1)
                        seg_cnt += 1
                    else:
                        curr = states[i]
                        seg_cnt += 1
                        segments.append([s,i])
                        midboundaries.append([i,len(segments)-1])
                        s = i
        if s>-1:
            segments.append([s, len(states)])

        merged = []
        for bd, segi in midboundaries:
            if segi >= len(segments)-1:
                continue
            if cossim(states[segments[segi][0]:segments[segi][1]].mean(0),
                      states[segments[segi+1][0]:segments[segi+1][1]].mean(0))>=mergethreshold:
                segments[segi+1]=[segments[segi][0],segments[segi+1][1]]
                merged.append(segi)
                continue
            s = max(segments[segi][0],bd-max(1,(segments[segi][1]-segments[segi][0])//2))
            bd = min(segments[segi+1][1],bd+max(1,(segments[segi+1][1]-segments[segi+1][0])//2))
            prev_center = states[segments[segi][0]:segments[segi][1]].mean(0)
            next_center = states[segments[segi+1][0]:segments[segi+1][1]].mean(0)
            sim_prev = cossim(states[s:bd],prev_center[None,:])
            sim_next = cossim(states[s:bd],next_center[None,:])
            sim_sweep = [(sim_prev[:i].sum()+sim_next[i:].sum()) for i in range(0,bd-s)]
            opt_b = np.arange(s,bd)[np.argmax(sim_sweep)]
            segments[segi] = [segments[segi][0],opt_b]
            segments[segi+1] = [opt_b,segments[segi+1][1]]  
        
        segments = [segment for segi, segment in enumerate(segments) if segi not in merged]

        return np.array(segments)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-dir', type=str, default='/data/sls/scratch/limingw/data/LibriSpeech_vad')
    parser.add_argument('--file-list', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/km_dir_top2048/discrete_speech_train_clsattnbndkmeans/file_list.txt')
    parser.add_argument('--boundary-json', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/km_dir_top2048/discrete_speech_train_clsattnbndkmeans/meta_data.json')
    parser.add_argument('--out-path', default='/data/sls/scratch/limingw/data/LibriSpeech_vad/syllablelm/train_data_bnd_dict.pkl')
    parser.add_argument('--model_type', default='5.0Hz')
    parser.add_argument('--ckpt_prefix', default='./models/SylBoost_500Hz')
    parser.add_argument('--sylboost_ckpt', default='')
    parser.add_argument('--n-parts', type=int, default=1)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--delta', type=float, default=0.0019)
    parser.add_argument('--quantile', type=float, default=0.75)
    parser.add_argument('--merge_threshold', type=float, default=0.4)
    parser.add_argument('--norm_threshold', type=float, default=0.01)
    parser.add_argument('--save_features', action='store_true')
    args = parser.parse_args()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.wav_dir)

    frames_per_embedding = 160

    sylboost_reader = SylBoostFeatureReader(
        f'{args.ckpt_prefix}.pth' if not len(args.sylboost_ckpt) else args.sylboost_ckpt,
        f'{args.ckpt_prefix}_kmeans.npy',
        f'{args.ckpt_prefix}_agglom.npy',
        args.model_type,
#        delta=FULL_MODELS_DICT[args.model_type]['delta'],
#        quantile=FULL_MODELS_DICT[args.model_type]['quantile'],
        delta=args.delta,
        quantile=args.quantile,
        merge_threshold=args.merge_threshold,
        norm_threshold=args.norm_threshold,
    )

    wavs, bounds, fns = data_loader.get_data_buckeye(args.file_list, args.boundary_json, parent_dir=args.wav_dir, load_wav=False)

    n_wavs = len(wavs) // args.n_parts + 1
    start_idx = n_wavs * args.part
    end_idx = n_wavs * (args.part + 1)
    wavs = wavs[start_idx:end_idx]
    bounds = bounds[start_idx:end_idx]
    fns = fns[start_idx:end_idx]

    ref_bounds = []
    seg_bounds = []
    save_bounds_dict = {}
    save_feat_dict = {}
    for idx in tqdm(list(range(len(wavs)))):
        wav = torchaudio.load(str(wav_dir / fns[idx]))[0]
        outputs = sylboost_reader.forward(wav.cuda().half())
        if args.save_features:
            save_feat_dict[fns[idx]] = outputs['features'][0].detach().cpu().numpy() 
        segments = outputs['segments'][0]

        pred_boundaries = data_loader.get_bounds(segments)
        pred_boundaries = np.array(pred_boundaries) * 2
        audio_len = pred_boundaries.max() + 1
        
        if len(bounds[idx]) > 0:
            boundaries = np.array(data_loader.get_bounds(bounds[idx])) // frames_per_embedding
            if idx < 10:
                print('idx, boundaries, pred_boundaries:', idx, boundaries, pred_boundaries)
            boundaries = np.minimum(boundaries[1:-1], audio_len-1)
            ref_bound = np.zeros(audio_len)
            seg_bound = np.zeros(audio_len)
            ref_bound[boundaries] = 1
            seg_bound[pred_boundaries] = 1
            ref_bound[-1] = 1
            seg_bound[-1] = 1

            ref_bounds.append(ref_bound)
            seg_bounds.append(seg_bound)

            save_bounds_dict[fns[idx]] = {'ref_bound': ref_bound, 'seg_bound': seg_bound}
        else:
            seg_bound = np.zeros(audio_len)
            seg_bound[pred_boundaries] = 1
            seg_bound[-1] = 1
            seg_bounds.append(seg_bound)

            save_bounds_dict[val_fns[idx]] = {'ref_bound': ref_bound, 'seg_bound': seg_bound}

        with open(args.out_path, 'wb') as f:
            pickle.dump(save_bounds_dict, f)

    if args.save_features:
        with open(args.out_path+'_hidden_states.pkl', 'wb') as f:
            pickle.dump(save_feat_dict, f)

    #precision, recall, f = eval_segmentation.score_boundaries(ref_bounds, seg_bounds, 2)
    #os = eval_segmentation.get_os(precision, recall)*100
    #r_val = eval_segmentation.get_rvalue(precision, recall)*100
    #print("Final result:", precision*100, recall*100, f*100, os, r_val, flush=True)
