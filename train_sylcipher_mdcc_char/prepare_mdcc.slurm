#!/bin/bash
#SBATCH -J prepare_mdcc
#SBATCH -o logs/%j_prepare_mdcc.out
#SBATCH -e logs/%j_prepare_mdcc.err
#SBATCH -p 2080
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

part=${1:-0}
n_parts=${n_parts:-10}

feat_name=xeus
seg_name=sylber_bnd
l=18
topk=1024
model=${MODEL_ROOT}/xeus/xeus.pth
wav_root=${DATA_ROOT}/MDCC/dataset
text_root=${DATA_ROOT}/MDCC/dataset/transcription
mdcc=${MANIFEST_ROOT}/mdcc
mdcc_vad=${MANIFEST_ROOT}/mdcc_vad
mdcc_char=${MANIFEST_ROOT}/mdcc_vad_raw_character
rvad_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/rVADfast/src/rVADfast/legacy_files

stage=${stage:-0}
stop_stage=${stop_stage:-80}

# Character transcripts (jyutping-free hanzi units)
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    mkdir -p ${mdcc_char}
    for x in test train; do
        python utils/prepare_mdcc_raw_character.py \
            --tsv-path ${mdcc_vad}/$x.tsv \
            --text-path $text_root \
            --out-path ${mdcc_char}/${x}_gradseg_style.json \
            --bad-path ${mdcc_char}/${x}_bad_files.txt
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    for x in test train; do
        python utils/rvads.py -r $rvad_dir --part $part --n-parts $n_parts \
            < ${mdcc}/$x.tsv > ${mdcc}/${x}_${part}.vads
    done
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    python utils/combine_vad.py --in_path "${mdcc}/{}_{}.vads" --out_path "${mdcc}/{}.vads" --n_parts $n_parts
    mkdir -p ${mdcc_vad}/wavs
    for x in train test; do
        python utils/remove_silence.py --tsv ${mdcc}/$x.tsv --vads ${mdcc}/${x}.vads --out ${mdcc_vad}/wavs
    done
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    for x in train test; do
        python utils/convert_tsv_to_filelist.py --tsv ${mdcc_vad}/$x.tsv --filelist ${mdcc_vad}/${x}_file_list.txt
    done
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    bash create_data_dir_sylber.sh ${feat_name}_L$l $topk $seg_name
fi

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    output_dir=${mdcc_vad}/feat_${feat_name}_L$l
    conda activate espnet2
    zsh prepare_audio_xeus.sh ${mdcc_vad} $output_dir $model 512 $l 100
    conda deactivate
    conda activate "${CONDA_ENV}"
fi

if [ $stage -le 60 ] && [ $stop_stage -ge 60 ]; then
    cwd=$PWD
    cd "${SYLBER_ROOT}"
    mkdir -p ${mdcc_vad}/$seg_name
    for x in test train; do
        conda activate sylber
        python extract_boundary.py --wav-dir ${mdcc_vad}/wavs \
            --file-list ${mdcc_vad}/${x}_file_list.txt \
            --out-path ${mdcc_vad}/$seg_name/${x}_data_bnd_dict_${part}.pkl \
            --part $part --n-parts $n_parts
        python combine_pkl.py \
            --in_path ${mdcc_vad}/$seg_name/${x}_data_bnd_dict_{}.pkl \
            --out_path ${mdcc_vad}/$seg_name/${x}_data_bnd_dict.json \
            --n_parts $n_parts
        conda deactivate

        conda activate "${CONDA_ENV}"
        python json_to_pkl.py \
            --in_path ${mdcc_vad}/$seg_name/${x}_data_bnd_dict.json \
            --out_path ${mdcc_vad}/$seg_name/${x}_data_bnd_dict.pkl
    done
    cd $cwd
fi

# Character-level k-means tree reuses the phonetic MDCC Sylber features
if [ $stage -le 70 ] && [ $stop_stage -ge 70 ]; then
    bash prepare_sylber_features_sls.sh \
        ${mdcc_char}/feat_${feat_name}_L${l} \
        ${mdcc_char}/feat_${feat_name}_L${l}_$seg_name \
        $topk
fi

if [ $stage -le 80 ] && [ $stop_stage -ge 80 ]; then
    bash prepare_text_sls.sh ${mdcc_char}/feat_${feat_name}_L${l}_$seg_name $topk False
fi
