import numpy as np
import torchaudio
import os,sys
import tqdm
bound_file=sys.argv[1] # word boundaries, 
                       # each line is written as:
                       # filename_path speaker_id word1_start,word1_end ... wordN_start,wordN_end 
                       # e.g: miscellaneous/dpparse/wolof
output_dir=sys.argv[2] # output directory
val_percent=0.1 # 0, 0.05 or 0.1, share of the validation dataset
                # will be chosen randomly among the speaker_ids
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

min_vad_dur=0.2 # does not work with longer dur
max_vad_len=19
units={}
dur={'value':[],'fid':[]}
concat={}
c=0
with open(bound_file) as buf:   
    for line in buf:
        data=line.rstrip().split(' ')
        fid,sid=data[:2]
        if len(data) < 3:
            continue
        if fid not in units: 
            units[fid]=[]
            concat[fid]=[]
        if ',' in data[2]:
            words=data[2:]
        else:
            words=[data[2]+','+data[3]]
        for word in words:
            s,e=word.split(',')
            s,e=float(s),float(e)
            if len(units[fid])>=1:
                # assert units are sorted along time
                assert s>=units[fid][-1][1],(fid,s,e,units[fid][-1])
            else:
                # adding the first silence as a word
                sil_dur=min(1,s) # if first silence is too long
                units[fid].append((s-sil_dur,s))
            units[fid].append((s,e))

for fid in tqdm.tqdm(units):
    vad_s,prev_e=units[fid][0]
    vad=[vad_s]
    for s,e in units[fid]:
        next_dur=(e-s)+(vad[-1]-vad[0])
        if next_dur>=max_vad_len:
            concat[fid].append(vad)
            vad=[s]
        elif s>prev_e+min_vad_dur: # if silence between tokens
            sil_dur=min(1,(s-np.around(prev_e,2))/2) # if silence is very long, take max 1 second
            vad.append(prev_e+sil_dur)
            concat[fid].append(vad)
            vad=[s-sil_dur,s]
        elif s>(prev_e+0.03):# if very small silence between tokens
            vad.append(s)
        prev_e=e
        vad.append(e)
    if len(vad)!=0:
        wav_path=fid
        assert os.path.isfile(wav_path), wav_path 
        wav,sr=torchaudio.load(wav_path)
        wav_end=np.around(len(wav[0])/sr,2)
        dur['value'].append(wav_end)
        dur['fid'].append(fid)
        if wav_end>vad[-1]:
            
            if wav_end-vad[-1]>1:
                wav_end=vad[-1]+1
            vad.append(wav_end)
        else:
            #if a segment ends after the end of the wav file
            ii=len(np.where(vad>=wav_end)[0])
            vad=vad[:-ii]
            vad.append(wav_end)
        concat[fid].append(vad)
# create valid set
valid=[]
if val_percent>0:
    indices=np.argsort(dur['value'])
    i=0
    valid_dur=0
    valid_total_dur=np.sum(dur['value'])*val_percent
    while valid_dur<valid_total_dur:
        valid.append(dur['fid'][indices[i]])
        valid_dur+=dur['value'][indices[i]]
        i+=1
    
    print(valid,valid_dur,valid_total_dur) 
    data={'train':[],'valid':[],'all':[]}
else:
    data={'all':[]}

max_vad=0         
for fid in concat:
    if fid in valid:
        subset='valid'
    else:
        subset='train'
    for vad in concat[fid]:
        vad=np.around(vad,2)
        bounds=vad[1:-1]
        if len(bounds)==0:
            continue
        assert (vad<=vad[-1]).all(),(bounds,vad,fid)
        c+=len(bounds)
        bounds=' '.join([str(t) for t in bounds])
        wav_path=fid 
        assert os.path.isfile(wav_path), wav_path 
        if val_percent>0:
            data[subset].append(' '.join([wav_path,'|',str(vad[0]),str(vad[-1]),'|',bounds]))
        data['all'].append(' '.join([wav_path,'|',str(vad[0]),str(vad[-1]),'|',bounds]))

for subset in data:
    with open(os.path.join(output_dir,subset+'.tsv'),'w') as buf:
        buf.write('\n'.join(data[subset])+'\n')

