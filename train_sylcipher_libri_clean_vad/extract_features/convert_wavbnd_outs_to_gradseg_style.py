import argparse  # TODO
import os
import pickle
import numpy as np
import torch

## from eval_segmentation.py at vqwordseg
def boundaries_to_intervals(boundaries):
    intervals = []
    j_prev = 0
    for j in np.where(boundaries)[0]:
        intervals.append((j_prev, j + 1))
        j_prev = j + 1
    return intervals


def intervals_to_boundaries(intervals):
    boundaries = np.zeros(intervals[-1][1], dtype=bool)
    boundaries[[i[1] - 1 for i in intervals]] = True
    return boundaries

def score_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the segmentation boundaries.
    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.
    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_boundaries_ref = 0
    n_boundaries_seg = 0
    n_boundaries_correct = 0
    for i_boundary, boundary_ref in enumerate(ref):
        boundary_seg = seg[i_boundary]
        assert boundary_ref[-1]  # check if last boundary is True
        assert boundary_seg[-1]
        
        # If lengths are the same, disregard last True reference boundary
        if len(boundary_ref) == len(boundary_seg):
            boundary_ref = boundary_ref[:-1]
            # boundary_seg = boundary_seg[:-1]

        boundary_seg = seg[i_boundary][:-1]  # last boundary is always True,
                                             # don't want to count this

        # If reference is longer, truncate
        if len(boundary_ref) > len(boundary_seg):
            boundary_ref = boundary_ref[:len(boundary_seg)]
        
        boundary_ref = list(np.nonzero(boundary_ref)[0])
        boundary_seg = list(np.nonzero(boundary_seg)[0])
        n_boundaries_ref += len(boundary_ref)
        n_boundaries_seg += len(boundary_seg)

        for i_seg in boundary_seg:
            for i, i_ref in enumerate(boundary_ref):
                if abs(i_seg - i_ref) <= tolerance:
                    n_boundaries_correct += 1
                    boundary_ref.pop(i)
                    break

    # Temp
#     print("n_boundaries_correct", n_boundaries_correct)
#     print("n_boundaries_seg", n_boundaries_seg)
#     print("n_boundaries_ref", n_boundaries_ref)

    precision = float(n_boundaries_correct)/n_boundaries_seg
    recall = float(n_boundaries_correct)/n_boundaries_ref
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf

    return precision, recall, f




def score_word_token_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the word token boundaries.
    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.
    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_tokens_ref = 0
    n_tokens_seg = 0
    n_tokens_correct = 0
    for i_boundary, boundary_ref in enumerate(ref):
        boundary_seg = seg[i_boundary]
        assert boundary_ref[-1]  # check if last boundary is True
        assert boundary_seg[-1]
        
        # The code below shouldn't be done for token scores
        # # If lengths are the same, disregard last True reference boundary
        # if len(boundary_ref) == len(boundary_seg):
        #     boundary_ref = boundary_ref[:-1]
        # boundary_seg = seg[i_boundary][:-1]  # last boundary is always True,
                                             # don't want to count this

        # If reference is longer, truncate
        if len(boundary_ref) > len(boundary_seg):
            boundary_ref = boundary_ref[:len(boundary_seg)]
            boundary_ref[-1] = True

        # Build list of ((word_start_lower, word_start_upper), (word_end_lower,
        # word_end_upper))
        word_bound_intervals = []
        for word_start, word_end in boundaries_to_intervals(boundary_ref):
            word_bound_intervals.append((
                (max(0, word_start - tolerance), word_start + tolerance),
                (word_end - tolerance, word_end + tolerance)
                ))
        seg_intervals = boundaries_to_intervals(boundary_seg)

        n_tokens_ref += len(word_bound_intervals)
        n_tokens_seg += len(seg_intervals)

        # Score word token boundaries
        for seg_start, seg_end in seg_intervals:
            # print seg_start, seg_end
            for i_gt_word, (word_start_interval,
                    word_end_interval) in enumerate(word_bound_intervals):
                word_start_lower, word_start_upper = word_start_interval
                word_end_lower, word_end_upper = word_end_interval

                if (word_start_lower <= seg_start <= word_start_upper and
                        word_end_lower <= seg_end <= word_end_upper):
                    n_tokens_correct += 1
                    word_bound_intervals.pop(i_gt_word)  # can't re-use token
                    # print "correct"
                    break

    # # Temp
    # print("n_tokens_correct", n_tokens_correct)
    # print("n_tokens_seg", n_tokens_seg)
    # print("n_tokens_ref", n_tokens_ref)

    precision = float(n_tokens_correct)/n_tokens_seg
    recall = float(n_tokens_correct)/n_tokens_ref
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf

    return precision, recall, f


def get_os(precision, recall):
    """Calculate over segmentation score."""
    if precision == 0:
        return -np.inf
    else:
        return recall/precision - 1


def get_rvalue(precision, recall):
    """Calculate the R-value."""
    os = get_os(precision, recall)
    r1 = np.sqrt((1 - recall)**2 + os**2)
    r2 = (-os + recall - 1)/np.sqrt(2)
    rvalue = 1 - (np.abs(r1) + np.abs(r2))/2
    return rvalue


parser = argparse.ArgumentParser()
parser.add_argument('--word_boundaries_txt', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/wav2boundaries/spokencoco/dataset/train_wav2boundaries_gradseg_unsup_unpaired_text_2048_oov_e2e_refinement/{}_prom_0.2_0.33_1_real/all_word_boundaries.txt')
parser.add_argument('--backoff_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylber_bnd')
parser.add_argument('--length_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/feat/hubert_large_ll60k_sylber_bnd/top_2048/km_dir_2048/discrete_speech_{}_clsattnbndkmeans')
parser.add_argument('--output_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/wav2boundaries/spokencoco/wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/')
parser.add_argument('--splits', default='val2,train')
args = parser.parse_args()

## in case prediction error on some utterances, back off into this instead
manifest_path =  args.backoff_dir
## where to save the new boundaries
new_manifest_path = args.output_dir 

which_sets = args.splits.split(',')
for which_set in which_sets:
    length_dict = {}
    word_boundaries_txt = args.word_boundaries_txt.format(which_set) 

    ## the data directory prepared for training a JSTTI model
    length_dir = args.length_dir.format(which_set)
    
    with open(os.path.join(length_dir, f'file_list.txt'), 'r') as f:
        file_lines = f.readlines()

    with open(os.path.join(length_dir, f'data.lengths'), 'r') as f:
        length_lines = f.readlines()

    for file_line, length_line in zip(file_lines, length_lines):
        length_dict[file_line.strip()] = int(length_line.strip())

    with open(word_boundaries_txt, 'r') as f:
        word_boundary_lines = f.readlines()

    # with open('/nobackup/users/junruin2/wav2boundaries/librispeech-clean-vad/dataset/train_gradseg_unsup/missing_prom_0.22_0.25_1_real/all_word_boundaries.txt', 'r') as f:
    with open(args.word_boundaries_txt.format('missing'), 'r') as f:
        word_boundary_lines.extend(f.readlines())

    print(len(word_boundary_lines))
    print(word_boundary_lines[-20:])
    filenames = []
    predicted_boundaries = []
    for word_boundary_line in word_boundary_lines:
        word_boundary_splitted = word_boundary_line.strip().split()
        # filename = 'train-clean' + word_boundary_splitted[0].split('train-clean')[1]
        filename = word_boundary_splitted[0]
        if filename in length_dict:
            filenames.append(filename)
            boundary_arr = np.zeros(length_dict[filename])
            for word_boundary_pairs in word_boundary_splitted[2:]:
                start, end = word_boundary_pairs.split(',')
                start = round(float(start) * 50)
                end = round(float(end) * 50)

                if start < length_dict[filename]:
                    boundary_arr[start] = 1
                elif start > length_dict[filename] + 1:
                    print(start, length_dict[filename], filename)

                if end < length_dict[filename]:
                    boundary_arr[end] = 1
                elif end > length_dict[filename] + 1:
                    print(end, length_dict[filename], filename)
            # boundary_arr[-1] = 1
            boundary_arr[0] = 1
            predicted_boundaries.append(boundary_arr)


    predicted_boundaries_dict = {}

    print(len(filenames), len(predicted_boundaries))
    for fn, pred_bnd in zip(filenames, predicted_boundaries):
        predicted_boundaries_dict[fn] = pred_bnd
        

    print('len of predicted_boundaries_dict', len(predicted_boundaries_dict))

    os.makedirs(new_manifest_path, exist_ok = True)

    
    with open(os.path.join(manifest_path, f'{which_set}_data_bnd_dict.pkl'), 'rb') as fb:
        boundary_dict = pickle.load(fb)


    all_available_filenames = list(boundary_dict.keys())


    missing_list = list(set(['train-clean-360/8195/117382/8195-117382-0112.flac',
                        'train-clean-360/159/121891/159-121891-0014.flac',
                        'train-clean-360/159/121891/159-121891-0010.flac',
                        'train-clean-360/159/121891/159-121891-0001.flac',
                        'train-clean-360/159/121891/159-121891-0000.flac',
                        'train-clean-100/1447/130552/1447-130552-0001.flac',
                        'train-clean-360/159/121891/159-121891-0002.flac',
                        'train-clean-360/159/121891/159-121891-0003.flac',
                        'train-clean-360/159/121891/159-121891-0004.flac',
                        'train-clean-360/159/121891/159-121891-0005.flac',
                        'train-clean-360/159/121891/159-121891-0006.flac',
                        'train-clean-360/159/121891/159-121891-0007.flac',
                        'train-clean-360/159/121891/159-121891-0008.flac',
                        'train-clean-360/159/121891/159-121891-0009.flac',
                        'train-clean-360/159/121891/159-121891-0011.flac',
                        'train-clean-360/159/121891/159-121891-0012.flac',
                        'train-clean-360/159/121891/159-121891-0013.flac',
                        'train-clean-360/472/129994/472-129994-0036.flac',
                        'train-clean-360/8113/117384/8113-117384-0017.flac']))

    all_available_filenames.extend(missing_list)

    new_boundary_dict = {}

    ref_boundaries = []
    seg_boundaries = []

    all_available_filenames_set = set(all_available_filenames)

    for filename in length_dict.keys():
        if filename not in  all_available_filenames_set :
            print(filename, 'wtf')
            print(filename in boundary_dict)


    for fidx, filename in enumerate(all_available_filenames):
        if filename in length_dict:
            if filename in boundary_dict and len(boundary_dict[filename]['ref_bound']) != len(boundary_dict[filename]['seg_bound']):
                print(filename, 'wtf')


            if fidx % 1000 == 0:
                print(fidx)
            new_boundary_dict[filename] = {}

            # curr_pred_bnd = predicted_boundaries_dict[filename]

            if filename in predicted_boundaries_dict:
                curr_pred_bnd = predicted_boundaries_dict[filename]
            #     boundary_arr = np.zeros(len(curr_pred_bnd) * 2).astype('int32')
            # else:
            if filename in boundary_dict:
                boundary_arr = np.zeros(len(boundary_dict[filename]['ref_bound'])).astype('int32')
                boundary_arr = boundary_dict[filename]['ref_bound']
                boundary_arr[0] = 1
                boundaries = boundary_dict[filename]

        #         ref_boundary_locs = np.where(boundaries['ref_bound'] == 1)[0]

        #         for ridx in range(len(ref_boundary_locs)):
        #             if ref_boundary_locs[ridx] >= len(boundary_arr):
        #                 print(ref_boundary_locs[ridx], len(boundary_arr))
        #                 ref_boundary_locs[ridx] = len(boundary_arr) -1

        #         boundary_arr[ref_boundary_locs] = 1
        #         boundary_arr[0] = 1

                new_boundary_dict[filename]['ref_bound'] =  boundary_arr


                # boundary_arr = np.zeros(len(curr_pred_bnd) * 2).astype('int32')
                
                boundaries = boundary_dict[filename]
                boundary_arr = np.zeros(len(boundary_dict[filename]['ref_bound'])).astype('int32')
                

                if filename not in predicted_boundaries_dict:
        #             seg_boundary_locs = np.where(boundaries['seg_bound'] == 1)[0]

        #             for ridx in range(len(seg_boundary_locs)):
        #                 if seg_boundary_locs[ridx] >= len(boundary_arr):
        #                     print(seg_boundary_locs[ridx], len(boundary_arr), filename)
        #                     seg_boundary_locs[ridx] = len(boundary_arr) -1

        #             boundary_arr[seg_boundary_locs] = 1
        #             boundary_arr[0] = 1
                    boundary_arr = boundaries['seg_bound']
                    boundary_arr[0] = 1
                else:
                    new_pred_bnds = predicted_boundaries_dict[filename]
                    seg_boundary_locs = np.where(new_pred_bnds == 1)[0] * 2
                    boundary_arr[seg_boundary_locs] = 1
                    boundary_arr[0] = 1


                new_boundary_dict[filename]['seg_bound'] = boundary_arr

            else:
                if filename in length_dict:
                    boundary_arr = np.zeros(length_dict[filename] * 2).astype('int32')
                    boundary_arr[0] = 1
                    new_pred_bnds = predicted_boundaries_dict[filename]
                    seg_boundary_locs = np.where(new_pred_bnds == 1)[0] * 2
                    boundary_arr[seg_boundary_locs] = 1
                    boundary_arr[0] = 1
                    new_boundary_dict[filename]['seg_bound'] = boundary_arr
                    new_boundary_dict[filename]['ref_bound'] = boundary_arr


    print('len of new_boundary_dict', len(new_boundary_dict), len(length_dict))

    with open(os.path.join(new_manifest_path, f'{which_set}_utts.pkl'), 'wb') as fb:
        pickle.dump(new_boundary_dict, fb) 
