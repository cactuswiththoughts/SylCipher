import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--in_path')
parser.add_argument('--out_path')
parser.add_argument('--n_parts', type=int)
args = parser.parse_args()

new_data_dict = {}
total = 0
for i in range(args.n_parts):
    with open(args.in_path.format(i), 'rb') as f:
        d = pkl.load(f)
        new_data_dict.update(d)
        total += len(d)

if args.out_path.endswith('.txt'):
    with open(args.out_path, 'w') as f_out,\
        open(args.out_path.replace('.txt', '.tsv'), 'w') as f_tsv:
        for fn, units in new_data_dict.items():
            sent = ' '.join(map(str, units[0].tolist()))
            print(f'{fn}\t{sent}', file=f_out)
            print(fn, file=f_tsv)
else:
    with open(args.out_path, 'wb') as f:
        pkl.dump(new_data_dict, f)

print(f'Total number of entries: {total}')
