from pathlib import Path
import argparse
import os


def _manifest_root():
    return Path(os.environ.get('MANIFEST_ROOT', Path(__file__).resolve().parents[3] / 'manifest'))


parser = argparse.ArgumentParser(description='Placeholder for merging per-language km_dirs into a shared codebook tree.')
mroot = _manifest_root()
parser.add_argument(
    '--src-dirs', nargs='+',
    default=[
        str(mroot / 'aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048'),
        str(mroot / 'mdcc_vad/feat_xeus_L18_sylber_bnd/top_608/km_dir_608'),
        str(mroot / 'suisiann_vad/feat_xeus_L18_sylber_ft_bnd/top_1024/km_dir_1024'),
    ],
)
parser.add_argument(
    '--out-dir',
    default=str(mroot / 'aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048'),
)
parser.add_argument('--shared-token', action='store_true', default=True)
args = parser.parse_args()

manifest_dirs = [Path(p) for p in args.src_dirs]
out_dir = Path(args.out_dir)
shared_token = args.shared_token

# Merge file lists, bad file lists and boundary dictionaries
# Merge text token dictionary
# Merge speech token dictionary
# Merge segment-level and frame-level centroids
print('src:', *[str(p) for p in manifest_dirs], sep='\n  ')
print('out:', out_dir)
print('shared_token:', shared_token)
print('Use combine_weights.py for the checkpoint / codebook merge.')
