import numpy as np
import shutil
import os
import json
from collections import defaultdict
NONWORD = ('<unk>')

topk = 1024
feat_name = 'xeus_L18'
bnd_name = 'sylber_ft_stage2_bnd'

write_dir = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_{feat_name}_{bnd_name}/top_{topk}'
os.makedirs(os.path.join(write_dir, f'km_dir_{topk}'), exist_ok =True)

dict_path = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/syllable_{topk}.txt'

if os.path.exists(dict_path):
    with open(dict_path, 'r') as f:
        lines = f.readlines()

    words = {}
    for line in lines:
        splitted = line.strip().split()
        words[splitted[0]] = int(splitted[1])
    
    top_words = sorted(words, key=lambda x:words[x], reverse=True) 
else:
    words = defaultdict(int)
    for which_set in ['train', 'test']:
        meta_path = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad/{which_set}_gradseg_style_with_tone.json'
        with open(meta_path, 'r') as f:
            meta_data_dict = json.load(f)
            for fn, segs in meta_data_dict.items():
                for _, _, w in segs:
                    if w not in NONWORD:
                        words[w] += 1

    top_words = sorted(words, key=lambda x:words[x], reverse=True)[:topk]
    with open(dict_path, 'w') as f_d:
        for w in top_words:
            print(f'{w} {words[w]}', file=f_d)

words['<OOV>'] = 0

for which_set in ['train', 'test']:
    meta_path = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad/{which_set}_gradseg_style_with_tone.json'
    with open(meta_path, 'r') as f:
        meta_data_dict = json.load(f)

    file_list_fn = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_{feat_name}_{bnd_name}/top_{topk}/km_dir_{topk}/discrete_speech_{which_set}_clsattnbndkmeans/file_list.txt'
    with open(file_list_fn, 'r') as f:
        file_lines = f.readlines()

    new_meta_path = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad/{which_set}_gradseg_style_with_tone_oov_{topk}.json'
    new_meta_data_dict = {}
    with open(os.path.join(write_dir, f'{which_set}_words.txt'), 'w') as fw:
        for file_line in file_lines:
            file_line = file_line.strip()
            to_write_list = []
            new_meta_data_dict[file_line] = []
            if file_line in meta_data_dict:
                for start, end, c in meta_data_dict[file_line]:
                    if c in top_words:
                        to_write_list.append(c)
                        new_meta_data_dict[file_line].append([start, end, c])
                    else:
                        to_write_list.append('<OOV>')
                        new_meta_data_dict[file_line].append([start, end, '<OOV>'])
                        words['<OOV>'] += 1

                fw.write(" ".join(to_write_list) + '\n')
            else:
                print(f'{file_line} not in meta_dict')
                fw.write('<OOV>\n')

    with open(new_meta_path, 'w') as fm:
        json.dump(new_meta_data_dict, fm, indent=2)

with open(os.path.join(write_dir,'dict.txt'), 'w') as fw:
    for word in words.keys():
        fw.write(word + ' ' + str(words[word]) + '\n')

shutil.copy(os.path.join(write_dir,'dict.txt'), os.path.join(write_dir,f'km_dir_{topk}','dict.txt'))
print(f'Replace {words["<OOV>"]} of {sum(words.values())} syllables with <OOV>')
