import torch

ckpt_path = '/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip_v2/checkpoint_best.pt'
state_dict = torch.load(ckpt_path)

encoder_keys = [k for k in state_dict['model'] if k.startswith('encoder.') and k not in ['encoder.src_proj.weight', 'encoder.src_proj.bias', 'encoder.proj.weight', 'encoder.proj.bias']]
for k in encoder_keys:
    print(f'{k}: {state_dict["model"][k].shape}')
encoder_dict = {k.replace('encoder.', ''):state_dict['model'][k] for k in encoder_keys}

out_path = '/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip_v2/encoder.pt'
torch.save(encoder_dict, out_path)
