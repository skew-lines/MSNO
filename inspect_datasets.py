import argparse
import csv
import os
from collections import Counter, defaultdict


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}
META_HINTS = ("meta", "protocol", "label", "trial", "trl", "csv", "txt")
SPLIT_NAMES = {"train", "training", "dev", "val", "validation", "test", "testing", "eval"}
CLASS_NAMES = {"real", "fake", "bonafide", "bona-fide", "spoof"}


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only dataset structure inspector")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    reports = []

    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        reports.append(inspect_dataset(path, root, args.max_depth, args.max_examples))

    text = "\n\n".join(format_report(report) for report in reports)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        write_summary_csv(args.output + ".csv", reports)


def inspect_dataset(dataset_dir, root, max_depth, max_examples):
    ext_counter = Counter()
    split_dirs = []
    class_dirs = []
    meta_files = []
    audio_examples = []
    files_by_parent = Counter()
    audio_by_top = Counter()

    dataset_depth = dataset_dir.rstrip(os.sep).count(os.sep)
    for current_dir, dirnames, filenames in os.walk(dataset_dir):
        rel_dir = os.path.relpath(current_dir, dataset_dir)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue

        for dirname in dirnames:
            low = dirname.lower()
            rel_path = os.path.join(rel_dir, dirname) if rel_dir != "." else dirname
            if low in SPLIT_NAMES:
                split_dirs.append(rel_path)
            if low in CLASS_NAMES:
                class_dirs.append(rel_path)

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            full_path = os.path.join(current_dir, filename)
            rel_path = os.path.relpath(full_path, dataset_dir)
            files_by_parent[rel_dir] += 1

            if ext in AUDIO_EXTENSIONS:
                ext_counter[ext] += 1
                top = rel_path.split(os.sep)[0] if os.sep in rel_path else "."
                audio_by_top[top] += 1
                if len(audio_examples) < max_examples:
                    audio_examples.append(rel_path)
            elif looks_like_meta(filename):
                meta_files.append(summarize_meta_file(full_path, rel_path, max_examples))

    return {
        "name": os.path.basename(dataset_dir),
        "path": dataset_dir,
        "ext_counter": ext_counter,
        "split_dirs": sorted(set(split_dirs)),
        "class_dirs": sorted(set(class_dirs)),
        "meta_files": meta_files[:max_examples],
        "audio_examples": audio_examples,
        "top_audio_dirs": audio_by_top.most_common(max_examples),
        "busy_dirs": files_by_parent.most_common(max_examples),
    }


def looks_like_meta(filename):
    low = filename.lower()
    ext = os.path.splitext(low)[1]
    if ext not in {".csv", ".txt", ".tsv", ".lst"}:
        return False
    return any(hint in low for hint in META_HINTS)


def summarize_meta_file(path, rel_path, max_examples):
    summary = {
        "path": rel_path,
        "size": safe_size(path),
        "header": "",
        "examples": [],
    }
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = sniff_dialect(sample)
            reader = csv.reader(f, dialect)
            rows = []
            for idx, row in enumerate(reader):
                if idx == 0:
                    summary["header"] = " | ".join(row[:12])
                elif len(rows) < max_examples:
                    rows.append(" | ".join(row[:12]))
                if idx >= max_examples:
                    break
            summary["examples"] = rows
    except Exception as exc:
        summary["header"] = f"<read error: {exc}>"
    return summary


def sniff_dialect(sample):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t |")
    except Exception:
        return csv.excel


def safe_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def format_report(report):
    lines = [
        f"## {report['name']}",
        f"path: {report['path']}",
        f"audio_ext_counts: {dict(report['ext_counter'])}",
        f"split_dirs: {report['split_dirs']}",
        f"class_dirs: {report['class_dirs']}",
        f"top_audio_dirs: {report['top_audio_dirs']}",
        f"busy_dirs: {report['busy_dirs']}",
        "audio_examples:",
    ]
    lines.extend(f"  - {item}" for item in report["audio_examples"])
    lines.append("meta_candidates:")
    for item in report["meta_files"]:
        lines.append(f"  - {item['path']} size={item['size']}")
        lines.append(f"    header: {item['header']}")
        for example in item["examples"][:3]:
            lines.append(f"    row: {example}")
    return "\n".join(lines)


def write_summary_csv(path, reports):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "audio_ext_counts", "split_dirs", "class_dirs", "meta_files"])
        for report in reports:
            writer.writerow([
                report["name"],
                dict(report["ext_counter"]),
                ";".join(report["split_dirs"]),
                ";".join(report["class_dirs"]),
                ";".join(item["path"] for item in report["meta_files"]),
            ])


if __name__ == "__main__":
    main()
