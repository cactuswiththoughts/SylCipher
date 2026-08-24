#!/usr/bin/env python3
"""
Prepare English text corpora for domain-mismatch UASR experiments.

Domains following arXiv:2110.03509:
  - librilm    : LibriSpeech LM text / literature
  - wiki       : Wikipedia / encyclopedia
  - newscrawl  : WMT NewsCrawl / news
  - imagec     : image captions = COCO captions + Conceptual Captions TSV

Examples:
  python prepare_domain_text.py \
      --out-dir data/domain_text \
      --domains librilm newscrawl imagec \
      --max-lines 300000 \
      --newscrawl-years 2014 \
      --cc-tsv /path/to/Train_GCC-training.tsv

  python prepare_domain_text.py \
      --out-dir data/domain_text \
      --domains wiki \
      --wiki-json-dir /path/to/wikiextractor_output \
      --max-lines 300000
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Optional


LIBRILM_URL = "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

WIKI_DUMP_URL = (
    "https://dumps.wikimedia.org/enwiki/latest/"
    "enwiki-latest-pages-articles-multistream.xml.bz2"
)

NEWCRAWL_URL_TMPL = (
    "https://data.statmt.org/news-crawl/en/"
    "news.{year}.en.shuffled.deduped.gz"
)


def download(url: str, path: Path, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"[skip] exists: {path}")
        return path

    print(f"[download] {url}")
    print(f"           -> {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def normalize_text(
    text: str,
    lowercase: bool = True,
    keep_digits: bool = False,
    keep_apostrophe: bool = True,
) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("`", "'").replace("“", " ").replace("”", " ")

    if lowercase:
        text = text.lower()

    if keep_digits:
        if keep_apostrophe:
            text = re.sub(r"[^a-z0-9']+", " ", text)
        else:
            text = re.sub(r"[^a-z0-9]+", " ", text)
    else:
        if keep_apostrophe:
            text = re.sub(r"[^a-z']+", " ", text)
        else:
            text = re.sub(r"[^a-z]+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    # Drop lines that are only apostrophes/spaces after normalization.
    if not re.search(r"[a-z]", text):
        return ""

    return text


def split_to_sentences(text: str) -> Iterator[str]:
    """
    Conservative sentence splitter. For NewsCrawl/LibriLM this mostly leaves
    one sentence per line. For WikiExtractor documents it breaks paragraphs.
    """
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    for p in parts:
        p = p.strip()
        if p:
            yield p


def filter_sentence(
    sent: str,
    min_words: int,
    max_words_per_line: int,
) -> bool:
    n = len(sent.split())
    return min_words <= n <= max_words_per_line


def write_stream(
    lines: Iterable[str],
    out_path: Path,
    *,
    max_lines: Optional[int],
    max_words: Optional[int],
    min_words: int,
    max_words_per_line: int,
    lowercase: bool,
    keep_digits: bool,
    keep_apostrophe: bool,
    dedup: bool,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_lines = 0
    n_words = 0
    n_seen_raw = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for raw in lines:
            n_seen_raw += 1
            norm = normalize_text(
                raw,
                lowercase=lowercase,
                keep_digits=keep_digits,
                keep_apostrophe=keep_apostrophe,
            )
            if not norm:
                continue
            if not filter_sentence(norm, min_words, max_words_per_line):
                continue
            if dedup:
                if norm in seen:
                    continue
                seen.add(norm)

            words = norm.split()
            fout.write(norm + "\n")
            n_lines += 1
            n_words += len(words)

            if max_lines is not None and n_lines >= max_lines:
                break
            if max_words is not None and n_words >= max_words:
                break

            if n_lines % 100000 == 0:
                print(f"  wrote {n_lines:,} lines / {n_words:,} words")

    stats = {
        "output": str(out_path),
        "raw_items_seen": n_seen_raw,
        "lines_written": n_lines,
        "words_written": n_words,
    }
    print(f"[done] {out_path}: {n_lines:,} lines, {n_words:,} words")
    return stats


def stream_librilm(raw_dir: Path, overwrite: bool) -> Iterator[str]:
    path = download(
        LIBRILM_URL,
        raw_dir / "librispeech-lm-norm.txt.gz",
        overwrite=overwrite,
    )
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def stream_newscrawl(raw_dir: Path, years: list[int], overwrite: bool) -> Iterator[str]:
    for year in years:
        url = NEWCRAWL_URL_TMPL.format(year=year)
        path = download(
            url,
            raw_dir / f"news.{year}.en.shuffled.deduped.gz",
            overwrite=overwrite,
        )
        with open_text(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def maybe_extract_wiki(
    raw_dir: Path,
    wiki_json_dir: Optional[Path],
    download_wiki: bool,
    overwrite: bool,
) -> Path:
    """
    Returns a WikiExtractor output directory.

    Requires:
      pip install wikiextractor

    This is intentionally opt-in because the English Wikipedia dump is huge.
    """
    if wiki_json_dir is not None:
        return wiki_json_dir

    if not download_wiki:
        raise ValueError(
            "For --domains wiki, provide --wiki-json-dir pointing to WikiExtractor "
            "output, or pass --download-wiki to download and extract enwiki."
        )

    dump_path = download(
        WIKI_DUMP_URL,
        raw_dir / "enwiki-latest-pages-articles-multistream.xml.bz2",
        overwrite=overwrite,
    )
    out_dir = raw_dir / "wikiextractor_enwiki"

    if out_dir.exists() and not overwrite:
        print(f"[skip] WikiExtractor output exists: {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "wikiextractor.WikiExtractor",
        str(dump_path),
        "-o",
        str(out_dir),
        "--json",
        "--processes",
        "8",
    ]
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd)
    return out_dir


def stream_wiki(
    raw_dir: Path,
    wiki_json_dir: Optional[Path],
    download_wiki: bool,
    overwrite: bool,
) -> Iterator[str]:
    wiki_dir = maybe_extract_wiki(
        raw_dir=raw_dir,
        wiki_json_dir=wiki_json_dir,
        download_wiki=download_wiki,
        overwrite=overwrite,
    )

    files = sorted([p for p in wiki_dir.rglob("*") if p.is_file()])
    if not files:
        raise RuntimeError(f"No WikiExtractor files found under {wiki_dir}")

    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # WikiExtractor --json gives one JSON document per line.
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    text = line

                for sent in split_to_sentences(text):
                    yield sent


def stream_coco(raw_dir: Path, overwrite: bool) -> Iterator[str]:
    zip_path = download(
        COCO_ANN_URL,
        raw_dir / "annotations_trainval2017.zip",
        overwrite=overwrite,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        wanted = [
            "annotations/captions_train2017.json",
            "annotations/captions_val2017.json",
        ]

        for name in wanted:
            if name not in names:
                print(f"[warn] missing from COCO zip: {name}")
                continue

            print(f"[read] {name}")
            with zf.open(name) as f:
                data = json.load(f)
            for ann in data.get("annotations", []):
                cap = ann.get("caption", "").strip()
                if cap:
                    yield cap


def stream_conceptual_captions(tsv_paths: list[Path]) -> Iterator[str]:
    """
    Conceptual Captions TSV format is usually:
      caption<TAB>image_url

    Download manually from Google's Conceptual Captions page and pass:
      --cc-tsv Train_GCC-training.tsv Validation_GCC-1.1.0-Validation.tsv
    """
    for path in tsv_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[read] Conceptual Captions TSV: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                caption = line.split("\t", 1)[0].strip()
                if caption:
                    yield caption


def stream_imagec(
    raw_dir: Path,
    cc_tsv: list[Path],
    overwrite: bool,
    include_coco: bool,
    include_cc: bool,
) -> Iterator[str]:
    if include_coco:
        yield from stream_coco(raw_dir, overwrite=overwrite)
    if include_cc:
        if not cc_tsv:
            print(
                "[warn] --domains imagec includes Conceptual Captions, but no "
                "--cc-tsv was provided. Output will contain COCO only."
            )
        else:
            yield from stream_conceptual_captions(cc_tsv)


def parse_years(s: str) -> list[int]:
    """
    Accepts:
      "2014"
      "2014,2015,2016"
      "2014-2018"
    """
    years = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["librilm", "wiki", "newscrawl", "imagec"],
        required=True,
    )

    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words-per-line", type=int, default=80)

    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--keep-digits", action="store_true")
    parser.add_argument("--drop-apostrophe", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")

    parser.add_argument(
        "--newscrawl-years",
        type=str,
        default="2014",
        help='Examples: "2014", "2014,2015", "2014-2018"',
    )

    parser.add_argument(
        "--wiki-json-dir",
        type=Path,
        default=None,
        help="Path to WikiExtractor --json output directory.",
    )
    parser.add_argument(
        "--download-wiki",
        action="store_true",
        help="Download enwiki dump and run WikiExtractor. Very large.",
    )

    parser.add_argument(
        "--cc-tsv",
        type=Path,
        nargs="*",
        default=[],
        help="Conceptual Captions TSV files, e.g. Train_GCC-training.tsv",
    )
    parser.add_argument("--no-coco", action="store_true")
    parser.add_argument("--no-conceptual-captions", action="store_true")

    args = parser.parse_args()

    out_dir: Path = args.out_dir
    raw_dir: Path = args.raw_dir or (out_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        max_lines=args.max_lines,
        max_words=args.max_words,
        min_words=args.min_words,
        max_words_per_line=args.max_words_per_line,
        lowercase=not args.no_lowercase,
        keep_digits=args.keep_digits,
        keep_apostrophe=not args.drop_apostrophe,
        dedup=not args.no_dedup,
    )

    all_stats = {}

    for domain in args.domains:
        print(f"\n=== Preparing domain: {domain} ===")
        out_path = out_dir / f"{domain}.txt"

        if domain == "librilm":
            lines = stream_librilm(raw_dir, overwrite=args.overwrite)

        elif domain == "newscrawl":
            years = parse_years(args.newscrawl_years)
            lines = stream_newscrawl(raw_dir, years=years, overwrite=args.overwrite)

        elif domain == "wiki":
            lines = stream_wiki(
                raw_dir=raw_dir,
                wiki_json_dir=args.wiki_json_dir,
                download_wiki=args.download_wiki,
                overwrite=args.overwrite,
            )

        elif domain == "imagec":
            lines = stream_imagec(
                raw_dir=raw_dir,
                cc_tsv=args.cc_tsv,
                overwrite=args.overwrite,
                include_coco=not args.no_coco,
                include_cc=not args.no_conceptual_captions,
            )

        else:
            raise ValueError(domain)

        stats = write_stream(lines, out_path, **common)
        all_stats[domain] = stats

    stats_path = out_dir / "prepare_domain_text_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n[stats] {stats_path}")


if __name__ == "__main__":
    main()
