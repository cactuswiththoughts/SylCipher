import argparse
import os
import random
from collections import Counter


def load_dict(dict_path):
    word_counts = {}
    with open(dict_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word_counts[parts[0]] = int(parts[1]) if len(parts) > 1 else 0
    return word_counts


def get_vocab(word_counts):
    if "<OOV>" in word_counts:
        return {w for w in word_counts if w != "<OOV>"}
    return set(word_counts)


def read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def filter_line(line, vocab, oov_token="<OOV>"):
    tokens = []
    for token in line.split():
        if token in vocab:
            tokens.append(token)
        else:
            tokens.append(oov_token)
    return tokens


def sample_lines(lines, n, rng):
    if n <= len(lines):
        return rng.sample(lines, n)
    return [rng.choice(lines) for _ in range(n)]


def mix_lines(source1_lines, source2_lines, ratio, rng, use_all_source1=False):
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    if use_all_source1:
        n1 = len(source1_lines)
        n2 = int(round(n1 * (1.0 - ratio) / ratio))
    else:
        total = min(len(source1_lines), len(source2_lines))
        n1 = int(round(total * ratio))
        n2 = total - n1

    sampled1 = sample_lines(source1_lines, n1, rng)
    sampled2 = sample_lines(source2_lines, n2, rng)
    mixed = sampled1 + sampled2
    rng.shuffle(mixed)
    return mixed, n1, n2


def write_mixed_text(lines, vocab, output_path, oov_token="<OOV>"):
    token_counts = Counter()
    oov_count = 0
    with open(output_path, "w") as fw:
        for line in lines:
            tokens = filter_line(line, vocab, oov_token=oov_token)
            for token in tokens:
                token_counts[token] += 1
                if token == oov_token:
                    oov_count += 1
            fw.write(" ".join(tokens) + "\n")
    return token_counts, oov_count


def write_dict(word_counts, token_counts, output_path, oov_token="<OOV>"):
    with open(output_path, "w") as fw:
        for word in word_counts:
            if word == oov_token:
                continue
            fw.write(f"{word} {token_counts.get(word, 0)}\n")
        if oov_token in word_counts:
            fw.write(f"{oov_token} {token_counts.get(oov_token, 0)}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Mix two train_words.txt sources and constrain tokens to a fixed "
            "dictionary so vocabulary size stays approximately the same."
        )
    )
    parser.add_argument("--source1", required=True, help="Path to first train_words.txt")
    parser.add_argument("--source2", required=True, help="Path to second train_words.txt")
    parser.add_argument("--dict_path", required=True, help="Fixed dictionary file")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for mixed train_words.txt",
    )
    parser.add_argument(
        "--output_dict",
        default=None,
        help="Output path for updated dict.txt (defaults to <output_dir>/dict.txt)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Fraction of utterances taken from source1 (default: 0.5)",
    )
    parser.add_argument(
        "--use_all_source1",
        action="store_true",
        help=(
            "Use all utterances from source1 and subsample source2 to match "
            "the requested ratio."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    word_counts = load_dict(args.dict_path)
    vocab = get_vocab(word_counts)
    oov_token = "<OOV>" if "<OOV>" in word_counts else None

    source1_lines = read_lines(args.source1)
    source2_lines = read_lines(args.source2)
    mixed_lines, n1, n2 = mix_lines(
        source1_lines,
        source2_lines,
        args.ratio,
        rng,
        use_all_source1=args.use_all_source1,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    token_counts, oov_count = write_mixed_text(
        mixed_lines,
        vocab,
        args.output,
        oov_token=oov_token or "<OOV>",
    )

    output_dict = args.output_dict
    if output_dict is None:
        output_dict = os.path.join(os.path.dirname(os.path.abspath(args.output)), "dict.txt")
    write_dict(word_counts, token_counts, output_dict, oov_token=oov_token or "<OOV>")

    total_tokens = sum(token_counts.values())
    print(f"source1 utterances used: {n1} / {len(source1_lines)}")
    print(f"source2 utterances used: {n2} / {len(source2_lines)}")
    print(f"mixed utterances written: {len(mixed_lines)}")
    print(f"dictionary size: {len(word_counts)}")
    print(f"total tokens: {total_tokens}")
    if oov_token is not None:
        print(
            f"replaced {oov_count} of {total_tokens} tokens with {oov_token} "
            f"({100.0 * oov_count / max(total_tokens, 1):.2f}%)"
        )
    print(f"wrote mixed text to {args.output}")
    print(f"wrote dictionary to {output_dict}")


if __name__ == "__main__":
    main()
