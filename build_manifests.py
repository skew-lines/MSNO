import argparse
import csv
import os
import random
from collections import Counter

import pandas as pd


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
FIELDNAMES = ["sample_name", "is_fake", "audio_path", "source", "split", "language", "attack"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build unified meta.csv manifests without moving audio files")
    parser.add_argument("--raw_root", type=str, default="/root/autodl-tmp")
    parser.add_argument("--out_root", type=str, default="/root/autodl-tmp/continual_manifests")
    parser.add_argument("--max_per_class", type=int, default=0, help="0 means keep all samples")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_large_external", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    raw_root = os.path.abspath(args.raw_root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    builders = [
        ("ASVspoof2019_LA", build_asvspoof2019_la),
        ("for-norm", build_for_norm),
        ("XMAD", build_xmad),
        ("ASVspoof5", build_asvspoof5),
        ("release_in_the_wild", build_release_in_the_wild),
        ("CodecFake", build_codecfake),
        ("ASVspoof2021_LA_eval", build_asvspoof2021_la_eval),
        ("ASVspoof2021_DF_eval", build_asvspoof2021_df_eval),
    ]
    if not args.skip_large_external:
        builders.append(("MLAAD_M-AILABS", build_mlaad_mailabs))

    for name, builder in builders:
        try:
            counts = builder(raw_root, out_root, args.max_per_class)
            print(f"[ok] {name}: {counts}")
        except FileNotFoundError as exc:
            print(f"[skip] {name}: missing {exc}")
        except Exception as exc:
            print(f"[error] {name}: {exc}")

    print("[note] ASVspoof2021 LA/DF eval require trial_metadata.txt with labels.")


def build_asvspoof2019_la(raw_root, out_root, max_per_class):
    root = os.path.join(raw_root, "ASVspoof2019_LA")
    train_protocol = os.path.join(root, "ASVspoof2019.LA.cm.train.trn.txt")
    eval_protocol = os.path.join(root, "ASVspoof2019.LA.cm.eval.trl.txt")

    train_rows = parse_asv_protocol(
        train_protocol,
        audio_dir=os.path.join(root, "train", "flac"),
        source="ASVspoof2019_LA",
        split="train",
    )
    test_rows = parse_asv_protocol(
        eval_protocol,
        audio_dir=os.path.join(root, "eval", "flac"),
        source="ASVspoof2019_LA",
        split="test",
    )
    train_rows = limit_per_class(train_rows, max_per_class)
    test_rows = limit_per_class(test_rows, max_per_class)
    return write_task(out_root, "ASVspoof2019_LA", {"train": train_rows, "test": test_rows})


def parse_asv_protocol(protocol_path, audio_dir, source, split):
    rows = []
    with open(protocol_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            attack = parts[3]
            label = parts[4].lower()
            rows.append(make_row(
                sample_name=utt_id,
                is_fake=0 if label == "bonafide" else 1,
                audio_path=os.path.join(audio_dir, utt_id + ".flac"),
                source=source,
                split=split,
                attack=attack,
            ))
    return rows


def build_for_norm(raw_root, out_root, max_per_class):
    root = os.path.join(raw_root, "for-norm")
    train_rows = collect_class_dirs(
        base_dir=os.path.join(root, "training"),
        source="for-norm",
        split="train",
    )
    test_rows = collect_class_dirs(
        base_dir=os.path.join(root, "testing"),
        source="for-norm",
        split="test",
    )
    train_rows = limit_per_class(train_rows, max_per_class)
    test_rows = limit_per_class(test_rows, max_per_class)
    return write_task(out_root, "for-norm", {"train": train_rows, "test": test_rows})


def build_xmad(raw_root, out_root, max_per_class):
    root = os.path.join(raw_root, "XMAD")
    train_rows = []
    test_rows = []
    for language in sorted(os.listdir(root)):
        language_dir = os.path.join(root, language)
        if not os.path.isdir(language_dir):
            continue
        train_meta = find_meta_file(os.path.join(language_dir, "train"))
        test_meta = find_meta_file(os.path.join(language_dir, "test"))
        if os.path.exists(train_meta):
            train_rows.extend(parse_xmad_meta(train_meta, language_dir, language, "train"))
        if os.path.exists(test_meta):
            test_rows.extend(parse_xmad_meta(test_meta, language_dir, language, "test"))

    train_rows = limit_per_class(train_rows, max_per_class)
    test_rows = limit_per_class(test_rows, max_per_class)
    return write_task(out_root, "XMAD", {"train": train_rows, "test": test_rows})


def parse_xmad_meta(meta_path, language_dir, language, split):
    rows = []
    missing = 0
    meta = read_table(meta_path)
    meta.columns = [str(column).strip().lstrip("\ufeff") for column in meta.columns]
    if "is_fake" not in meta.columns:
        raise KeyError(f"{meta_path} columns={list(meta.columns)}")
    sample_column = "sample_name" if "sample_name" in meta.columns else meta.columns[0]

    for _, row in meta.iterrows():
        label = int(row["is_fake"])
        sample_name = str(row[sample_column])
        audio_path = resolve_xmad_audio_path(
            os.path.join(language_dir, split, "fake" if label == 1 else "real"),
            sample_name,
        )
        if audio_path is None:
            missing += 1
            continue
        rows.append(make_row(
            sample_name=sample_name,
            is_fake=label,
            audio_path=audio_path,
            source="XMAD",
            split=split,
            language=language,
            attack=str(row.get("meta", "")),
        ))
    if missing:
        print(f"[warn] XMAD {language}/{split}: skipped {missing} rows with missing audio in {meta_path}")
    return rows


def resolve_xmad_audio_path(class_dir, sample_name):
    candidates = [
        os.path.join(class_dir, sample_name),
        os.path.join(class_dir, sample_name.replace(":", "_")),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def build_asvspoof5(raw_root, out_root, max_per_class):
    root = first_existing([
        os.path.join(raw_root, "ASVspoof5"),
        os.path.join(raw_root, "Asvspoof5"),
    ])
    if root is None:
        raise FileNotFoundError("ASVspoof5")
    train_protocol = first_existing([
        os.path.join(root, "ASVspoof5.train.tsv"),
        os.path.join(root, "ASVspoof5.train.track_2.trial.tsv"),
        os.path.join(root, "ASVspoof5.training.track_2.trial.tsv"),
    ])
    eval_protocol = first_existing([
        os.path.join(root, "ASVspoof5.eval.track_1.tsv"),
        os.path.join(root, "ASVspoof5.eval.track_2.trial.tsv"),
    ])

    if train_protocol is None:
        raise FileNotFoundError("ASVspoof5 train protocol not found. Need ASVspoof5.train.tsv")
    if eval_protocol is None:
        raise FileNotFoundError("ASVspoof5 eval protocol not found")

    train_rows = parse_asvspoof5_protocol(
        train_protocol,
        audio_dir=os.path.join(root, "flac_T"),
        source="ASVspoof5",
        split="train",
    )
    test_rows = parse_asvspoof5_protocol(
        eval_protocol,
        audio_dir=os.path.join(root, "flac_E_eval"),
        source="ASVspoof5",
        split="test",
    )
    train_rows = limit_per_class(train_rows, max_per_class)
    test_rows = limit_per_class(test_rows, max_per_class)
    return write_task(out_root, "ASVspoof5", {"train": train_rows, "test": test_rows})


def parse_asvspoof5_protocol(protocol_path, audio_dir, source, split):
    rows = []
    with open(protocol_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            utt_id = parts[1]

            label_token = find_label_token(parts)
            if label_token is None:
                continue

            attack = find_attack_token(parts)
            is_fake = 0 if label_token in {"bonafide", "bona-fide", "real", "target", "nontarget"} else 1
            rows.append(make_row(
                sample_name=utt_id,
                is_fake=is_fake,
                audio_path=os.path.join(audio_dir, utt_id + ".flac"),
                source=source,
                split=split,
                attack=attack,
            ))
    return rows


def build_asvspoof2021_la_eval(raw_root, out_root, max_per_class):
    root = os.path.join(raw_root, "ASVspoof2021_LA_eval")
    rows = parse_asvspoof2021_metadata(
        metadata_path=os.path.join(root, "trial_metadata.txt"),
        audio_dir=os.path.join(root, "flac"),
        source="ASVspoof2021_LA_eval",
    )
    rows = limit_per_class(rows, max_per_class)
    return write_task(out_root, "ASVspoof2021_LA_eval", {"test": rows})


def build_asvspoof2021_df_eval(raw_root, out_root, max_per_class):
    root = os.path.join(raw_root, "ASVspoof2021_DF_eval")
    rows = parse_asvspoof2021_metadata(
        metadata_path=os.path.join(root, "trial_metadata.txt"),
        audio_dir=os.path.join(root, "flac"),
        source="ASVspoof2021_DF_eval",
    )
    rows = limit_per_class(rows, max_per_class)
    return write_task(out_root, "ASVspoof2021_DF_eval", {"test": rows})


def parse_asvspoof2021_metadata(metadata_path, audio_dir, source):
    rows = []
    with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            utt_id = find_token_with_prefix(parts, ("LA_E_", "DF_E_"))
            if utt_id is None:
                utt_id = first_audio_like_id(parts)
            label_token = find_label_token(parts)
            if utt_id is None or label_token is None:
                continue

            attack = find_attack_token(parts)
            rows.append(make_row(
                sample_name=utt_id,
                is_fake=0 if label_token in {"bonafide", "bona-fide", "real"} else 1,
                audio_path=os.path.join(audio_dir, utt_id + ".flac"),
                source=source,
                split="test",
                attack=attack,
            ))
    if not rows:
        raise ValueError(f"No labeled rows parsed from {metadata_path}")
    return rows


def build_release_in_the_wild(raw_root, out_root, max_per_class):
    root = os.path.join(raw_root, "release_in_the_wild")
    meta_path = os.path.join(root, "meta.csv")
    rows = []
    with open(meta_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            sample_name = row["file"]
            rows.append(make_row(
                sample_name=sample_name,
                is_fake=0 if label in {"bonafide", "bona-fide", "real"} else 1,
                audio_path=os.path.join(root, sample_name),
                source="release_in_the_wild",
                split="test",
            ))
    rows = limit_per_class(rows, max_per_class)
    return write_task(out_root, "release_in_the_wild", {"test": rows})


def build_codecfake(raw_root, out_root, max_per_class):
    root = os.path.join(raw_root, "CodecFake")
    rows = collect_class_dirs(root, source="CodecFake", split="test")
    rows = limit_per_class(rows, max_per_class)
    return write_task(out_root, "CodecFake", {"test": rows})


def build_mlaad_mailabs(raw_root, out_root, max_per_class):
    mlaad_root = first_existing([
        os.path.join(raw_root, "MLAAD"),
        os.path.join(raw_root, "MLADD"),
    ])
    mailabs_root = os.path.join(raw_root, "M-AILABS")
    if mlaad_root is None:
        raise FileNotFoundError("MLAAD or MLADD")
    if not os.path.isdir(mailabs_root):
        raise FileNotFoundError(mailabs_root)

    rows = []
    rows.extend(collect_audio_tree(mlaad_root, is_fake=1, source="MLAAD", split="test"))
    rows.extend(collect_audio_tree(mailabs_root, is_fake=0, source="M-AILABS", split="test"))
    rows = limit_per_class(rows, max_per_class)
    return write_task(out_root, "MLAAD_M-AILABS", {"test": rows})


def collect_class_dirs(base_dir, source, split):
    rows = []
    for class_name, label in (("real", 0), ("fake", 1)):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        rows.extend(collect_audio_tree(class_dir, is_fake=label, source=source, split=split, class_root=class_dir))
    if not rows:
        raise FileNotFoundError(base_dir)
    return rows


def collect_audio_tree(root, is_fake, source, split, class_root=None):
    rows = []
    base = class_root or root
    for current_dir, _, filenames in os.walk(root):
        for filename in filenames:
            if not is_audio(filename):
                continue
            path = os.path.join(current_dir, filename)
            rel_name = os.path.relpath(path, base)
            language = infer_language_from_path(path, root)
            attack = infer_attack_from_path(path, root)
            rows.append(make_row(
                sample_name=rel_name,
                is_fake=is_fake,
                audio_path=path,
                source=source,
                split=split,
                language=language,
                attack=attack,
            ))
    return rows


def make_row(sample_name, is_fake, audio_path, source, split, language="", attack=""):
    return {
        "sample_name": sample_name,
        "is_fake": int(is_fake),
        "audio_path": os.path.abspath(audio_path),
        "source": source,
        "split": split,
        "language": language,
        "attack": attack,
    }


def write_task(out_root, task_name, split_rows):
    counts = {}
    for split, rows in split_rows.items():
        out_dir = os.path.join(out_root, task_name, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "meta.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        counts[split] = count_labels(rows)
    return counts


def count_labels(rows):
    counter = Counter(row["is_fake"] for row in rows)
    return {"real": counter.get(0, 0), "fake": counter.get(1, 0), "total": len(rows)}


def limit_per_class(rows, max_per_class):
    if max_per_class <= 0:
        return rows
    rows = rows[:]
    random.shuffle(rows)
    kept = []
    counts = Counter()
    for row in rows:
        label = row["is_fake"]
        if counts[label] < max_per_class:
            kept.append(row)
            counts[label] += 1
    return kept


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def find_meta_file(split_dir):
    preferred = [
        os.path.join(split_dir, "meta.txt"),
        os.path.join(split_dir, "meta.csv"),
    ]
    found = first_existing(preferred)
    if found is None:
        return os.path.join(split_dir, "meta.csv")
    return found


def read_table(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def find_token_with_prefix(tokens, prefixes):
    for token in tokens:
        if any(token.startswith(prefix) for prefix in prefixes):
            return token
    return None


def find_label_token(tokens):
    labels = {
        "bonafide",
        "bona-fide",
        "real",
        "spoof",
        "fake",
        "target",
        "nontarget",
    }
    for token in reversed(tokens):
        normalized = token.strip().lower()
        if normalized in labels:
            return normalized
    return None


def find_attack_token(tokens):
    for token in tokens:
        normalized = token.strip()
        if len(normalized) >= 2 and normalized[0].upper() == "A" and normalized[1:].isdigit():
            return normalized
    return ""


def first_audio_like_id(tokens):
    for token in tokens:
        if token.lower().endswith(AUDIO_EXTENSIONS):
            return os.path.splitext(os.path.basename(token))[0]
    for token in tokens:
        if "_E_" in token or token.startswith(("LA_", "DF_", "T_", "D_", "E_")):
            return os.path.splitext(os.path.basename(token))[0]
    return None


def is_audio(filename):
    return os.path.splitext(filename)[1].lower() in AUDIO_EXTENSIONS


def infer_language_from_path(path, root):
    rel = os.path.relpath(path, root)
    first = rel.split(os.sep)[0]
    if len(first) in {2, 5} or "_" in first:
        return first
    return ""


def infer_attack_from_path(path, root):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[1]
    return ""


if __name__ == "__main__":
    main()
