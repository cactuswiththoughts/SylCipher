in_path="/nobackup/users/limingw/data/LibriSpeech_vad/lm_syllable_pyphen_plus_oov_2048.txt"

tot_count = 0
oov_count = 0
OOV = '<OOV>'
with open(in_path, 'r') as f:
    for l in f:
        for u in l.split():
            if u == OOV:
                oov_count += 1
            tot_count += 1

print(f'{oov_count / tot_count} ({oov_count}/{tot_count}) of units are OOV')
                 
