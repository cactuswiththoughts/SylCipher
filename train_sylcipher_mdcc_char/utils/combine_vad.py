import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/suisiann/{}_{}.vads')
parser.add_argument('--out_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/suisiann/{}.vads')
parser.add_argument('--n_parts', type=int, default=10)

args = parser.parse_args()

n_parts = args.n_parts
for x in ['train', 'test']:
    lines = []
    n_files = 0
    out_path = args.out_path.format(x)
    with open(out_path, 'w') as f_out:
        for i in range(n_parts):
            in_path = args.in_path.format(x, i)
            with open(in_path, 'r') as f:
                lines_i = f.read().strip().split('\n')
                lines.extend(lines_i)
                n_files += len(lines_i)

        print('\n'.join(lines), file=f_out)
        print(f'Number of {x} files: {n_files}')
