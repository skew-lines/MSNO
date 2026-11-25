import argparse
import os
from collections import Counter

import pandas as pd


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
BAD_SAMPLE_VALUES = {"", "#NAME?", "#VALUE!", "NAN", "NONE", "NULL"}


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether meta.csv rows point to existing audio files")
    parser.add_argument("--meta", type=str, required=True, help="Path to the original meta.csv")
    parser.add_argument("--split_dir", type=str, required=True, help="Directory containing real/ and fake/ folders")
    parser.add_argument("--max_examples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.meta, sep=None, engine="python", dtype=str)
    df.columns = [str(column).strip().lstrip("\ufeff") for column in df.columns]

    sample_col = "sample_name" if "sample_name" in df.columns else df.columns[0]
    if "is_fake" not in df.columns:
        raise KeyError(f"Missing is_fake column in {args.meta}. columns={list(df.columns)}")

    stats = Counter()
    bad_meta_examples = []
    missing_examples = []

    for row_idx, row in df.iterrows():
        sample = normalize_value(row.get(sample_col))
        label = normalize_value(row.get("is_fake"))

        if sample.upper() in BAD_SAMPLE_VALUES:
            stats["bad_meta"] += 1
            add_example(bad_meta_examples, args.max_examples, row_idx, sample, label, "bad sample_name")
            continue

        try:
            label_int = int(float(label))
        except Exception:
            stats["bad_meta"] += 1
            add_example(bad_meta_examples, args.max_examples, row_idx, sample, label, "bad is_fake")
            continue

        class_dir = "fake" if label_int == 1 else "real"
        expected_path = os.path.join(args.split_dir, class_dir, sample)
        existing_path = resolve_audio_path(expected_path)
        if existing_path is None:
            stats["missing_audio"] += 1
            add_example(missing_examples, args.max_examples, row_idx, sample, label, expected_path)
        else:
            stats["ok"] += 1

    print(f"meta: {args.meta}")
    print(f"split_dir: {args.split_dir}")
    print(f"columns: {list(df.columns)}")
    print(f"sample_column: {sample_col}")
    print(f"total_rows: {len(df)}")
    print(f"ok: {stats['ok']}")
    print(f"bad_meta: {stats['bad_meta']}")
    print(f"missing_audio: {stats['missing_audio']}")

    print_examples("bad_meta_examples", bad_meta_examples)
    print_examples("missing_audio_examples", missing_examples)


def normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def resolve_audio_path(path):
    path_variants = [path, path.replace(":", "_")]
    for path_variant in path_variants:
        if os.path.exists(path_variant):
            return path_variant

    root, ext = os.path.splitext(path)
    if ext:
        return None

    for audio_ext in AUDIO_EXTENSIONS:
        for root_variant in (root, root.replace(":", "_")):
            candidate = root_variant + audio_ext
            if os.path.exists(candidate):
                return candidate
    return None


def add_example(examples, max_examples, row_idx, sample, label, detail):
    if len(examples) < max_examples:
        examples.append({
            "row": row_idx,
            "sample": sample,
            "is_fake": label,
            "detail": detail,
        })


def print_examples(title, examples):
    print(f"\n[{title}]")
    if not examples:
        print("none")
        return

    for item in examples:
        print(
            f"row={item['row']} sample={item['sample']} "
            f"is_fake={item['is_fake']} detail={item['detail']}"
        )


if __name__ == "__main__":
    main()
