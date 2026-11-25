import argparse
import copy
import csv
import gc
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from UnifiedDataset import UnifiedDataset
from cl_baselines import build_strategy
from model import Model


# =========================================================
# Arguments
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paper-faithful continual learning baselines "
            "for XLS-R audio deepfake detection"
        )
    )

    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ssl_checkpoint",
        type=str,
        default="/root/autodl-fs/xlsr2_300m.pt",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--start_task_idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume_after_task_idx",
        type=int,
        default=None,
        help=(
            "Checkpoints created by this version also restore RAWM/RWM/RegO "
            "strategy state and the RAWM classifier teacher. Other strategy states such as "
            "ER/GEM memory, EWC Fisher, SI/MAS omega, and the LwF teacher are not "
            "restored; for paper-grade runs of those methods, prefer starting "
            "from scratch."
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train_tasks",
        type=str,
        default="ASVspoof2019_LA,for-norm,XMAD,ASVspoof5",
    )

    parser.add_argument(
        "--external_tasks",
        type=str,
        default=(
            "release_in_the_wild,"
            "CodecFake,"
            "ASVspoof2021_LA_eval,"
            "ASVspoof2021_DF_eval,"
            "MLADD_M-AILABS"
        ),
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Deprecated alias for --train_tasks.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/baselines",
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--num_epochs_per_task",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--balanced_ce",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--disable_balanced_ce",
        action="store_false",
        dest="balanced_ce",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--freeze_ssl",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--unfreeze_ssl",
        action="store_false",
        dest="freeze_ssl",
    )

    parser.add_argument(
        "--memory_size",
        type=int,
        default=1000,
        help=(
            "Total memory for ER/A-GEM; "
            "also used to derive GEM memory_per_task."
        ),
    )

    parser.add_argument(
        "--memory_per_task",
        type=int,
        default=None,
        help="Per-task episodic memory size for original GEM.",
    )

    parser.add_argument(
        "--replay_batch_size",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--replay_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--importance_batches",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--si_lambda",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--si_eps",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--mas_lambda",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lwf_alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lwf_temperature",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--wm_alpha",
        type=float,
        default=0.1,
        help="RLS regularizer for RAWM/RWM classifier projection.",
    )

    parser.add_argument(
        "--wm_eps",
        type=float,
        default=1e-6,
        help="Numerical floor used by RAWM/RWM.",
    )

    parser.add_argument(
        "--rawm_m",
        type=float,
        default=0.1,
        help="RAWM complementary-direction scale.",
    )

    parser.add_argument(
        "--rawm_eta",
        type=float,
        default=0.5,
        help="RAWM balance between projected CE and historical KD gradients.",
    )

    parser.add_argument(
        "--rawm_temperature",
        type=float,
        default=2.0,
        help="RAWM historical-classifier distillation temperature.",
    )

    parser.add_argument(
        "--rwm_score_temperature",
        type=float,
        default=1.0,
        help="Temperature for repository-compatible RWM batch scores.",
    )

    parser.add_argument(
        "--rego_ef_threshold",
        type=float,
        default=0.1,
        help="Ebbinghaus retention threshold used by RegO.",
    )

    parser.add_argument(
        "--rego_importance_quantile",
        type=float,
        default=0.75,
        help="Per-class Fisher quantile used to select important RegO regions.",
    )

    parser.add_argument(
        "--rego_eps",
        type=float,
        default=1e-12,
        help="Numerical floor for RegO gradient projections.",
    )

    parser.add_argument(
        "--gem_margin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--gem_qp_max_iter",
        type=int,
        default=200,
    )

    return parser.parse_args()


# =========================================================
# Method name
# =========================================================
def normalize_method_name(method: str) -> str:
    method = method.lower()

    aliases = {
        "finetuning": "fine_tuning",
        "naive": "fine_tuning",
        "sequential": "fine_tuning",
        "replay": "er",
        "a-gem": "agem",
    }

    return aliases.get(method, method)


# =========================================================
# Dataset / DataLoader
# =========================================================
def build_dataset(
    task_name,
    args,
    split,
    return_path=False,
):
    """
    单独创建 Dataset。

    这样训练和 after_task 可以共享同一个 Dataset，
    避免同一个 task 重复 read_csv() / 创建 meta。
    """

    return UnifiedDataset(
        args=args,
        data_dir=os.path.join(
            args.database_path,
            task_name,
        ),
        split=split,
        return_path=return_path,
    )


def build_loader(
    dataset,
    args,
    split,
    shuffle=False,
    drop_last=False,
):
    """
    基于已经创建好的 Dataset 构造 DataLoader。

    persistent_workers=False:
        每次 DataLoader iterator 生命周期结束后，
        worker 不会被永久保留。
    """

    return DataLoader(
        dataset,
        batch_size=(
            args.batch_size
            if split == "train"
            else args.eval_batch_size
        ),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def get_loader(
    task_name,
    args,
    split,
    shuffle=False,
    drop_last=False,
    return_path=False,
):
    """
    用于 evaluation 的便捷接口。

    Eval Dataset 不需要跨阶段共享，因此这里直接创建。
    """

    dataset = build_dataset(
        task_name,
        args,
        split,
        return_path=return_path,
    )

    return build_loader(
        dataset,
        args,
        split,
        shuffle=shuffle,
        drop_last=drop_last,
    )


# =========================================================
# Class weights
# =========================================================
def compute_class_weights(
    task_name,
    args,
    device,
):
    meta_path = os.path.join(
        args.database_path,
        task_name,
        "train",
        "meta.csv",
    )

    meta = pd.read_csv(meta_path)

    counts = (
        meta["is_fake"]
        .astype(int)
        .value_counts()
    )

    real_count = int(
        counts.get(0, 0)
    )

    fake_count = int(
        counts.get(1, 0)
    )

    # 这里 meta 后面已经不需要
    del meta
    gc.collect()

    if (
        real_count == 0
        or fake_count == 0
    ):
        raise ValueError(
            f"Both classes are required for {task_name}: "
            f"real={real_count}, "
            f"fake={fake_count}"
        )

    total = (
        real_count
        + fake_count
    )

    weights = torch.tensor(
        [
            total / (2.0 * real_count),
            total / (2.0 * fake_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[*] Class weights for task={task_name}: "
        f"real_count={real_count}, "
        f"fake_count={fake_count}, "
        f"weight_real={weights[0].item():.6f}, "
        f"weight_fake={weights[1].item():.6f}"
    )

    return weights


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    model,
    strategy,
    loader,
    optimizer,
    device,
    desc,
):
    """
    返回：
        avg_total_loss
        avg_ce_loss
        avg_acc

    对于 LwF：

        strategy.loss
            = CE + alpha * KD

        ce_loss
            = 当前任务纯 CE

    backward 始终根据 strategy.loss。

    LwF early stopping 只根据 ce_loss。
    """

    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0

    total_correct = 0
    total_seen = 0

    pbar = tqdm(
        loader,
        desc=desc,
        leave=True,
    )

    for x, y in pbar:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.long().to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        # -------------------------------------------------
        # 纯 CE：
        # 只用于统计 / LwF early stopping。
        #
        # detach logits，避免为 monitor CE 保留额外计算图。
        #
        # 使用 strategy.criterion，保证和 strategy 内部
        # balanced CE 权重完全一致。
        # -------------------------------------------------
        with torch.no_grad():
            ce_loss = strategy.criterion(
                logits.detach(),
                y,
            )

        # -------------------------------------------------
        # 真正用于训练的 objective
        #
        # LwF:
        #   CE + KD
        #
        # EWC:
        #   CE + EWC penalty
        #
        # ...
        # -------------------------------------------------
        loss = strategy.loss(
            x,
            y,
            logits,
        )

        strategy.backward(loss)

        strategy.after_backward()

        strategy.before_step()

        optimizer.step()

        strategy.after_step()

        batch_size = x.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_ce_loss += (
            ce_loss.item()
            * batch_size
        )

        total_correct += (
            logits.argmax(dim=1)
            == y
        ).sum().item()

        total_seen += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}",
            acc=(
                f"{100.0 * total_correct / max(total_seen, 1):.2f}%"
            ),
        )

    avg_total_loss = (
        total_loss
        / max(total_seen, 1)
    )

    avg_ce_loss = (
        total_ce_loss
        / max(total_seen, 1)
    )

    avg_acc = (
        100.0
        * total_correct
        / max(total_seen, 1)
    )

    return (
        avg_total_loss,
        avg_ce_loss,
        avg_acc,
    )


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    score_path=None,
    score_txt_path=None,
):
    model.eval()

    labels = []
    scores = []

    rows = []
    txt_rows = []

    sample_idx = 0

    for batch in tqdm(
        loader,
        desc="eval",
        leave=False,
    ):

        if len(batch) == 3:
            x, y, paths = batch
        else:
            x, y = batch

            paths = [
                str(i)
                for i in range(
                    sample_idx,
                    sample_idx + len(y),
                )
            ]

        x = x.to(
            device,
            non_blocking=True,
        )

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1,
        )

        prob_real = probs[:, 0]
        prob_fake = probs[:, 1]

        batch_labels = (
            y.cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        batch_scores = (
            prob_fake.cpu()
            .numpy()
            .tolist()
        )

        batch_real_scores = (
            prob_real.cpu()
            .numpy()
            .tolist()
        )

        batch_logits = (
            logits.cpu()
            .numpy()
            .tolist()
        )

        labels.extend(
            batch_labels
        )

        scores.extend(
            batch_scores
        )

        for (
            label,
            score_real,
            score_fake,
            logit,
        ) in zip(
            batch_labels,
            batch_real_scores,
            batch_scores,
            batch_logits,
        ):

            rows.append(
                [
                    sample_idx,
                    label,
                    float(logit[0]),
                    float(logit[1]),
                    float(score_real),
                    float(score_fake),
                ]
            )

            sample_idx += 1

        for (
            path,
            label,
            score_real,
            score_fake,
            logit,
        ) in zip(
            paths,
            batch_labels,
            batch_real_scores,
            batch_scores,
            batch_logits,
        ):

            txt_rows.append(
                [
                    path,
                    float(logit[0]),
                    float(logit[1]),
                    float(score_real),
                    float(score_fake),
                    label,
                ]
            )

    labels_np = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores_np = np.asarray(
        scores,
        dtype=np.float64,
    )

    if score_path is not None:

        os.makedirs(
            os.path.dirname(score_path),
            exist_ok=True,
        )

        with open(
            score_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "sample_idx",
                    "label",
                    "logit_real",
                    "logit_fake",
                    "score_real",
                    "score_fake",
                ]
            )

            writer.writerows(
                rows
            )

    if score_txt_path is not None:

        os.makedirs(
            os.path.dirname(
                score_txt_path
            ),
            exist_ok=True,
        )

        with open(
            score_txt_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
            )

            writer.writerow(
                [
                    "wav_path",
                    "logit_real",
                    "logit_fake",
                    "score_real",
                    "score_fake",
                    "y",
                ]
            )

            writer.writerows(
                txt_rows
            )

    return {
        "eer": compute_eer(
            labels_np,
            scores_np,
        ),
        "auc": compute_auc(
            labels_np,
            scores_np,
        ),
    }


# =========================================================
# Metrics
# =========================================================
def compute_auc(
    labels,
    scores,
):
    pos = scores[
        labels == 1
    ]

    neg = scores[
        labels == 0
    ]

    if (
        len(pos) == 0
        or len(neg) == 0
    ):
        return float("nan")

    order = np.argsort(
        scores
    )

    ranks = np.empty_like(
        order,
        dtype=np.float64,
    )

    ranks[order] = np.arange(
        1,
        len(scores) + 1,
    )

    pos_ranks = ranks[
        labels == 1
    ].sum()

    return float(
        (
            pos_ranks
            - len(pos)
            * (len(pos) + 1)
            / 2.0
        )
        /
        (
            len(pos)
            * len(neg)
        )
    )


def compute_eer(
    labels,
    scores,
):
    if (
        len(np.unique(labels))
        < 2
    ):
        return float("nan")

    order = np.argsort(
        scores
    )[::-1]

    labels = labels[
        order
    ]

    positives = np.sum(
        labels == 1
    )

    negatives = np.sum(
        labels == 0
    )

    fars = (
        np.cumsum(
            labels == 0
        )
        / max(
            negatives,
            1,
        )
    )

    frrs = (
        1.0
        - (
            np.cumsum(
                labels == 1
            )
            / max(
                positives,
                1,
            )
        )
    )

    idx = np.nanargmin(
        np.abs(
            fars - frrs
        )
    )

    return float(
        (
            fars[idx]
            + frrs[idx]
        )
        / 2.0
    )


def summarize_metrics(
    tasks,
    eer_matrix,
    auc_matrix,
):
    last_task = tasks[-1]

    final_eers = [
        eer_matrix[last_task][task]
        for task in tasks
    ]

    final_aucs = [
        auc_matrix[last_task][task]
        for task in tasks
    ]

    af_eer = []
    af_auc = []

    # 最后一个任务没有后续遗忘
    for i, task in enumerate(
        tasks[:-1]
    ):

        eer_history = []
        auc_history = []

        # 从任务学习完成后一直统计到最终模型
        for train_task in tasks[i:]:

            if (
                train_task in eer_matrix
                and task in eer_matrix[train_task]
            ):
                eer_history.append(
                    eer_matrix[
                        train_task
                    ][task]
                )

            if (
                train_task in auc_matrix
                and task in auc_matrix[train_task]
            ):
                auc_history.append(
                    auc_matrix[
                        train_task
                    ][task]
                )

        if (
            last_task not in eer_matrix
            or task not in eer_matrix[last_task]
            or not eer_history
        ):
            continue

        if (
            last_task not in auc_matrix
            or task not in auc_matrix[last_task]
            or not auc_history
        ):
            continue

        final_eer = (
            eer_matrix[
                last_task
            ][task]
        )

        final_auc = (
            auc_matrix[
                last_task
            ][task]
        )

        best_eer = np.nanmin(
            eer_history
        )

        best_auc = np.nanmax(
            auc_history
        )

        af_eer.append(
            final_eer
            - best_eer
        )

        af_auc.append(
            best_auc
            - final_auc
        )

    return {
        "final_avg_eer": float(
            np.nanmean(
                final_eers
            )
        ),

        "final_avg_auc": float(
            np.nanmean(
                final_aucs
            )
        ),

        "af_eer": (
            float(
                np.nanmean(
                    af_eer
                )
            )
            if af_eer
            else float("nan")
        ),

        "af_auc": (
            float(
                np.nanmean(
                    af_auc
                )
            )
            if af_auc
            else float("nan")
        ),
    }


# =========================================================
# Matrix
# =========================================================
def save_matrix(
    path,
    row_names,
    col_names,
    matrix,
):
    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "train_step"
            ]
            + col_names
        )

        for row in row_names:

            writer.writerow(
                [row]
                + [
                    matrix
                    .get(row, {})
                    .get(col, "")
                    for col
                    in col_names
                ]
            )


def load_matrix(
    path,
):
    matrix = {}

    if not os.path.exists(path):
        return matrix

    with open(
        path,
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            train_step = row.pop(
                "train_step"
            )

            matrix[
                train_step
            ] = {}

            for (
                task,
                value,
            ) in row.items():

                if value == "":
                    continue

                matrix[
                    train_step
                ][task] = float(
                    value
                )

    return matrix


# =========================================================
# External metrics
# =========================================================
def save_external_metrics(
    path,
    rows,
):
    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "train_step",
                "external_task",
                "eer",
                "auc",
                "score_path",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def load_external_metrics(
    path,
):
    rows = []

    if not os.path.exists(path):
        return rows

    with open(
        path,
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if (
                row.get(
                    "eer",
                    "",
                )
                != ""
            ):
                row[
                    "eer"
                ] = float(
                    row["eer"]
                )

            if (
                row.get(
                    "auc",
                    "",
                )
                != ""
            ):
                row[
                    "auc"
                ] = float(
                    row["auc"]
                )

            rows.append(
                row
            )

    return rows


# =========================================================
# Logger
# =========================================================
def log_message(
    log_file,
    message,
):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    message_with_time = (
        f"[{timestamp}] {message}"
    )

    print(
        message_with_time
    )

    with open(
        log_file,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            message_with_time
            + "\n"
        )


# =========================================================
# Model
# =========================================================
def build_model(
    args,
    device,
):
    model = Model(
        args,
        device,
    ).to(
        device
    )

    if args.model_path:

        state_dict = torch.load(
            args.model_path,
            map_location=device,
        )

        if (
            isinstance(
                state_dict,
                dict,
            )
            and "model"
            in state_dict
        ):
            state_dict = (
                state_dict[
                    "model"
                ]
            )

        cleaned = OrderedDict()

        for (
            key,
            value,
        ) in state_dict.items():

            cleaned[
                key[7:]
                if key.startswith(
                    "module."
                )
                else key
            ] = value

        missing, unexpected = (
            model.load_state_dict(
                cleaned,
                strict=False,
            )
        )

        print(
            f"[*] Loaded model_path="
            f"{args.model_path}"
        )

        print(
            f"[*] Missing keys: "
            f"{len(missing)}, "
            f"unexpected keys: "
            f"{len(unexpected)}"
        )

    if args.freeze_ssl:

        for param in (
            model
            .ssl_model
            .parameters()
        ):
            param.requires_grad = False

        print(
            "[*] Frozen XLS-R "
            "backbone parameters"
        )

    else:

        print(
            "[*] XLS-R backbone "
            "is trainable"
        )

    trainable = sum(
        param.numel()
        for param
        in model.parameters()
        if param.requires_grad
    )

    total = sum(
        param.numel()
        for param
        in model.parameters()
    )

    print(
        f"[*] Trainable parameters: "
        f"{trainable} / {total}"
    )

    return model


# =========================================================
# Main
# =========================================================
def main():

    args = parse_args()

    args.method = normalize_method_name(
        args.method
    )

    # =====================================================
    # LwF special early stopping
    #
    # LwF:
    #   patience = 3
    #   monitor = CE
    #
    # backward:
    #   CE + KD
    # =====================================================
    if args.method == "lwf":
        args.early_stop_patience = 3

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if args.tasks is not None:

        args.train_tasks = (
            args.tasks
        )

    train_tasks = [
        task.strip()
        for task
        in args.train_tasks.split(",")
        if task.strip()
    ]

    external_tasks = [
        task.strip()
        for task
        in args.external_tasks.split(",")
        if task.strip()
    ]

    run_dir = (
        args.run_dir
        if args.run_dir
        is not None
        else os.path.join(
            args.output_dir,
            args.method,
            f"seed_{args.seed}",
        )
    )

    log_dir = (
        args.log_dir
        if args.log_dir
        is not None
        else os.path.join(
            run_dir,
            "logs",
        )
    )

    os.makedirs(
        os.path.join(
            run_dir,
            "checkpoints",
        ),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            run_dir,
            "metrics",
        ),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            run_dir,
            "scores",
        ),
        exist_ok=True,
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    log_file = os.path.join(
        log_dir,
        "train.log",
    )

    train_epoch_path = os.path.join(
        run_dir,
        "metrics",
        "train_epochs.csv",
    )

    external_metrics_path = os.path.join(
        run_dir,
        "metrics",
        "external_metrics.csv",
    )

    # =====================================================
    # Save config
    # =====================================================
    with open(
        os.path.join(
            run_dir,
            "config.json",
        ),
        "w",
    ) as f:

        config = vars(
            args
        ).copy()

        config[
            "resolved_run_dir"
        ] = run_dir

        config[
            "resolved_log_dir"
        ] = log_dir

        config[
            "early_stop_monitor"
        ] = (
            "ce_loss"
            if args.method in {"lwf", "rawm"}
            else "total_loss"
        )

        json.dump(
            config,
            f,
            indent=2,
        )

    if not os.path.exists(
        log_file
    ):

        with open(
            log_file,
            "w",
            encoding="utf-8",
        ) as f:

            f.write("")

    # =====================================================
    # Epoch CSV
    # =====================================================
    if not os.path.exists(
        train_epoch_path
    ):

        with open(
            train_epoch_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "timestamp",
                    "task_idx",
                    "task",
                    "epoch",

                    "loss",
                    "ce_loss",
                    "acc",

                    "monitor_name",
                    "monitor_loss",
                    "best_monitor_loss",

                    "weight_real",
                    "weight_fake",

                    "bad_epochs",
                    "early_stopped",
                ]
            )

    log_message(
        log_file,
        f"[*] Run directory: {run_dir}",
    )

    log_message(
        log_file,
        f"[*] Log directory: {log_dir}",
    )

    log_message(
        log_file,
        f"[*] Device: {device}",
    )

    log_message(
        log_file,
        f"[*] Method: {args.method}",
    )

    log_message(
        log_file,
        f"[*] Train tasks: {train_tasks}",
    )

    log_message(
        log_file,
        f"[*] External tasks: {external_tasks}",
    )

    log_message(
        log_file,
        (
            f"[*] Early-stop patience="
            f"{args.early_stop_patience}"
        ),
    )

    if args.method in {"lwf", "rawm"}:

        log_message(
            log_file,
            "[*] Early-stop monitor=CE loss; "
            "backward objective includes distillation.",
        )

    else:

        log_message(
            log_file,
            "[*] Early-stop monitor=total strategy loss.",
        )

    if (
        args.resume_after_task_idx
        is not None
    ):

        log_message(
            log_file,
            "[warning] RAWM/RWM/RegO state is restored when present in the checkpoint. "
            "Other baseline strategy states are not yet checkpointed; rerun those "
            "methods from scratch for paper-grade experiments.",
        )

    eer_matrix = load_matrix(
        os.path.join(
            run_dir,
            "metrics",
            "eer_matrix.csv",
        )
    )

    auc_matrix = load_matrix(
        os.path.join(
            run_dir,
            "metrics",
            "auc_matrix.csv",
        )
    )

    external_rows = load_external_metrics(
        external_metrics_path
    )

    model = None
    strategy = None
    optimizer = None

    resume_eval_idx = (
        args.resume_after_task_idx
    )

    effective_start_idx = (
        args.start_task_idx
    )

    if resume_eval_idx is not None:

        effective_start_idx = (
            resume_eval_idx
        )

    # =====================================================
    # 这些方法的 after_task() 需要遍历 Dataset
    # =====================================================
    methods_need_after_task_loader = {
        "er",
        "ewc",
        "mas",
        "agem",
        "gem",
        "rego",
    }

    # =====================================================
    # Task loop
    # =====================================================
    for (
        task_idx,
        task_name,
    ) in enumerate(
        train_tasks
    ):

        if (
            task_idx
            < effective_start_idx
        ):

            log_message(
                log_file,
                f"[*] Skip completed "
                f"task={task_name} "
                f"idx={task_idx}",
            )

            continue

        # =================================================
        # Model
        # =================================================
        if model is None:

            model = build_model(
                args,
                device,
            )

            strategy = build_strategy(
                args.method,
                model,
                args,
                device,
            )

            if args.model_path:
                loaded_checkpoint = torch.load(
                    args.model_path,
                    map_location=device,
                )
                if (
                    isinstance(loaded_checkpoint, dict)
                    and loaded_checkpoint.get("strategy") is not None
                ):
                    strategy.load_state_dict(loaded_checkpoint["strategy"])
                    log_message(
                        log_file,
                        f"[*] Restored strategy state from: {args.model_path}",
                    )
                del loaded_checkpoint

            optimizer = torch.optim.Adam(
                filter(
                    lambda p: p.requires_grad,
                    model.parameters(),
                ),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        skip_training_for_resume_eval = (
            resume_eval_idx
            is not None
            and task_idx
            == resume_eval_idx
        )

        strategy.before_task(
            task_idx
        )

        # =================================================
        # Class weights
        # =================================================
        class_weights = None

        if (
            args.balanced_ce
            and not skip_training_for_resume_eval
        ):

            class_weights = (
                compute_class_weights(
                    task_name,
                    args,
                    device,
                )
            )

            strategy.set_class_weights(
                class_weights
            )

        else:

            strategy.set_class_weights(
                None
            )

        weight_real = (
            float(
                class_weights[0].item()
            )
            if class_weights
            is not None
            else ""
        )

        weight_fake = (
            float(
                class_weights[1].item()
            )
            if class_weights
            is not None
            else ""
        )

        log_message(
            log_file,
            f"[*] Start task={task_name} "
            f"idx={task_idx} "
            f"weight_real={weight_real} "
            f"weight_fake={weight_fake}",
        )

        best_monitor_loss = float(
            "inf"
        )

        best_total_loss = float(
            "inf"
        )

        best_model_state = None

        bad_epochs = 0

        # =================================================
        # Resume eval
        # =================================================
        if skip_training_for_resume_eval:

            log_message(
                log_file,
                f"[*] Resume eval after completed "
                f"task={task_name}; "
                f"skip training and checkpoint save.",
            )

        else:

            # =============================================
            # 只创建一次 train Dataset
            # =============================================
            train_dataset = build_dataset(
                task_name,
                args,
                split="train",
                return_path=False,
            )

            # =============================================
            # Training loader
            # =============================================
            train_loader = build_loader(
                train_dataset,
                args,
                split="train",
                shuffle=True,
                drop_last=True,
            )

            # =============================================
            # Epoch loop
            # =============================================
            for epoch in range(
                args.num_epochs_per_task
            ):

                (
                    loss,
                    ce_loss,
                    acc,
                ) = train_one_epoch(
                    model,
                    strategy,
                    train_loader,
                    optimizer,
                    device,
                    desc=(
                        f"{args.method} "
                        f"{task_name} "
                        f"epoch {epoch}"
                    ),
                )

                log_message(
                    log_file,
                    f"[train] "
                    f"task={task_name} "
                    f"epoch={epoch} "
                    f"loss={loss:.6f} "
                    f"ce_loss={ce_loss:.6f} "
                    f"acc={acc:.2f}",
                )

                # =========================================
                # Monitor
                # =========================================
                if args.method in {"lwf", "rawm"}:

                    monitor_loss = (
                        ce_loss
                    )

                    monitor_name = (
                        "ce_loss"
                    )

                else:

                    monitor_loss = (
                        loss
                    )

                    monitor_name = (
                        "train_loss"
                    )

                early_stopped = False

                # =========================================
                # Improvement
                # =========================================
                if (
                    monitor_loss
                    <
                    best_monitor_loss
                    - args.early_stop_min_delta
                ):

                    best_monitor_loss = (
                        monitor_loss
                    )

                    best_total_loss = (
                        loss
                    )

                    best_model_state = (
                        copy.deepcopy(
                            model.state_dict()
                        )
                    )

                    bad_epochs = 0

                    log_message(
                        log_file,
                        f"[early-stop] "
                        f"task={task_name} "
                        f"epoch={epoch} "
                        f"{monitor_name} improved "
                        f"to {best_monitor_loss:.6f}",
                    )

                # =========================================
                # No improvement
                # =========================================
                else:

                    bad_epochs += 1

                    log_message(
                        log_file,
                        f"[early-stop] "
                        f"task={task_name} "
                        f"epoch={epoch} "
                        f"{monitor_name}="
                        f"{monitor_loss:.6f} "
                        f"did not improve "
                        f"(best="
                        f"{best_monitor_loss:.6f}). "
                        f"bad_epochs="
                        f"{bad_epochs}/"
                        f"{args.early_stop_patience}",
                    )

                    if (
                        bad_epochs
                        >= args.early_stop_patience
                    ):

                        early_stopped = True

                        log_message(
                            log_file,
                            f"[early-stop] "
                            f"stop task={task_name} "
                            f"at epoch={epoch}, "
                            f"monitor="
                            f"{monitor_name}, "
                            f"best="
                            f"{best_monitor_loss:.6f}",
                        )

                # =========================================
                # Save epoch record
                # =========================================
                with open(
                    train_epoch_path,
                    "a",
                    newline="",
                ) as f:

                    writer = csv.writer(
                        f
                    )

                    writer.writerow(
                        [
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                            task_idx,
                            task_name,
                            epoch,

                            f"{loss:.8f}",
                            f"{ce_loss:.8f}",
                            f"{acc:.6f}",

                            monitor_name,
                            f"{monitor_loss:.8f}",
                            f"{best_monitor_loss:.8f}",

                            weight_real,
                            weight_fake,

                            bad_epochs,
                            int(
                                early_stopped
                            ),
                        ]
                    )

                if early_stopped:
                    break

            # =============================================
            # Training loader 已经没用了
            # =============================================
            del train_loader

            gc.collect()

            # =============================================
            # Restore best model
            # =============================================
            if (
                best_model_state
                is not None
            ):

                model.load_state_dict(
                    best_model_state
                )

                monitor_name = (
                    "ce_loss"
                    if args.method in {"lwf", "rawm"}
                    else "train_loss"
                )

                log_message(
                    log_file,
                    f"[*] Restored best model "
                    f"for task={task_name}, "
                    f"best_{monitor_name}="
                    f"{best_monitor_loss:.6f}, "
                    f"corresponding_total_loss="
                    f"{best_total_loss:.6f}",
                )

            # =============================================
            # after_task
            #
            # LwF / SI / Fine-tuning:
            #     不需要重新遍历 Dataset
            #
            # ER / EWC / MAS / AGEM / GEM:
            #     共用 train_dataset
            # =============================================
            if (
                args.method
                in methods_need_after_task_loader
            ):

                consolidate_loader = (
                    build_loader(
                        train_dataset,
                        args,
                        split="train",
                        shuffle=True,
                        drop_last=False,
                    )
                )

                strategy.after_task(
                    task_idx,
                    consolidate_loader,
                )

                # -----------------------------------------
                # 立即释放
                # -----------------------------------------
                del consolidate_loader

            else:

                strategy.after_task(
                    task_idx,
                    None,
                )

            # =============================================
            # 当前 task Dataset 生命周期结束
            # =============================================
            del train_dataset

            gc.collect()

            # =============================================
            # Save checkpoint
            # =============================================
            checkpoint_path = os.path.join(
                run_dir,
                "checkpoints",
                (
                    f"task_"
                    f"{task_idx + 1}_"
                    f"{task_name}_best.pt"
                ),
            )

            monitor_name = (
                "ce_loss"
                if args.method == "lwf"
                else "train_loss"
            )

            torch.save(
                {
                    "task_idx": (
                        task_idx
                    ),

                    "task_name": (
                        task_name
                    ),

                    "best_train_loss": (
                        best_total_loss
                    ),

                    "best_monitor_loss": (
                        best_monitor_loss
                    ),

                    "monitor_name": (
                        monitor_name
                    ),

                    "early_stop_patience": (
                        args.early_stop_patience
                    ),

                    "model": (
                        model.state_dict()
                    ),

                    "strategy": (
                        strategy.state_dict()
                    ),
                },
                checkpoint_path,
            )

            log_message(
                log_file,
                f"[*] Saved best checkpoint "
                f"to: {checkpoint_path}",
            )

            # best_model_state 是完整 state_dict copy
            # checkpoint 保存完成后已经不需要
            del best_model_state

            gc.collect()

        # =================================================
        # Internal evaluation
        # =================================================
        eer_matrix[
            task_name
        ] = {}

        auc_matrix[
            task_name
        ] = {}

        for eval_task in (
            train_tasks[
                : task_idx + 1
            ]
        ):

            eval_loader = get_loader(
                eval_task,
                args,
                "test",
                shuffle=False,
                drop_last=False,
                return_path=True,
            )

            score_path = os.path.join(
                run_dir,
                "scores",
                (
                    f"after_"
                    f"{task_idx + 1}_"
                    f"{task_name}"
                    f"__test_"
                    f"{eval_task}.csv"
                ),
            )

            score_txt_path = os.path.join(
                run_dir,
                "scores",
                (
                    f"after_"
                    f"{task_idx + 1}_"
                    f"{task_name}"
                    f"__test_"
                    f"{eval_task}"
                    f"_scores.txt"
                ),
            )

            metrics = evaluate(
                model,
                eval_loader,
                device,
                score_path=score_path,
                score_txt_path=score_txt_path,
            )

            eer_matrix[
                task_name
            ][eval_task] = (
                metrics["eer"]
            )

            auc_matrix[
                task_name
            ][eval_task] = (
                metrics["auc"]
            )

            log_message(
                log_file,
                f"[eval] "
                f"after={task_name} "
                f"test={eval_task} "
                f"EER="
                f"{metrics['eer']:.6f} "
                f"AUC="
                f"{metrics['auc']:.6f} "
                f"scores="
                f"{score_txt_path}",
            )

            # =============================================
            # Eval Dataset + DataLoader 立即释放
            # =============================================
            del eval_loader
            del metrics

            gc.collect()

        # =================================================
        # External evaluation
        # =================================================
        for external_task in (
            external_tasks
        ):

            eval_loader = get_loader(
                external_task,
                args,
                "test",
                shuffle=False,
                drop_last=False,
                return_path=True,
            )

            score_path = os.path.join(
                run_dir,
                "scores",
                (
                    f"after_"
                    f"{task_idx + 1}_"
                    f"{task_name}"
                    f"__external_"
                    f"{external_task}.csv"
                ),
            )

            score_txt_path = os.path.join(
                run_dir,
                "scores",
                (
                    f"after_"
                    f"{task_idx + 1}_"
                    f"{task_name}"
                    f"__external_"
                    f"{external_task}"
                    f"_scores.txt"
                ),
            )

            metrics = evaluate(
                model,
                eval_loader,
                device,
                score_path=score_path,
                score_txt_path=score_txt_path,
            )

            external_rows.append(
                {
                    "train_step": (
                        task_name
                    ),

                    "external_task": (
                        external_task
                    ),

                    "eer": (
                        metrics["eer"]
                    ),

                    "auc": (
                        metrics["auc"]
                    ),

                    "score_path": (
                        score_txt_path
                    ),
                }
            )

            log_message(
                log_file,
                f"[external] "
                f"after={task_name} "
                f"test={external_task} "
                f"EER="
                f"{metrics['eer']:.6f} "
                f"AUC="
                f"{metrics['auc']:.6f} "
                f"scores="
                f"{score_txt_path}",
            )

            # =============================================
            # External Dataset + DataLoader 立即释放
            # =============================================
            del eval_loader
            del metrics

            gc.collect()

        # =================================================
        # Save matrices
        # =================================================
        save_matrix(
            os.path.join(
                run_dir,
                "metrics",
                "eer_matrix.csv",
            ),
            train_tasks,
            train_tasks,
            eer_matrix,
        )

        save_matrix(
            os.path.join(
                run_dir,
                "metrics",
                "auc_matrix.csv",
            ),
            train_tasks,
            train_tasks,
            auc_matrix,
        )

        save_external_metrics(
            external_metrics_path,
            external_rows,
        )

        # =================================================
        # Task 级清理
        # =================================================
        gc.collect()

        log_message(
            log_file,
            f"[*] Finished task={task_name}; "
            f"released temporary Dataset/DataLoader references.",
        )

    # =====================================================
    # Final summary
    # =====================================================
    summary = summarize_metrics(
        train_tasks,
        eer_matrix,
        auc_matrix,
    )

    if external_rows:

        final_external = [
            row
            for row
            in external_rows
            if (
                row["train_step"]
                == train_tasks[-1]
            )
        ]

        if final_external:

            summary[
                "external_final_avg_eer"
            ] = float(
                np.nanmean(
                    [
                        float(
                            row["eer"]
                        )
                        for row
                        in final_external
                    ]
                )
            )

            summary[
                "external_final_avg_auc"
            ] = float(
                np.nanmean(
                    [
                        float(
                            row["auc"]
                        )
                        for row
                        in final_external
                    ]
                )
            )

        else:

            summary[
                "external_final_avg_eer"
            ] = float("nan")

            summary[
                "external_final_avg_auc"
            ] = float("nan")

    with open(
        os.path.join(
            run_dir,
            "metrics",
            "final_summary.json",
        ),
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    log_message(
        log_file,
        json.dumps(
            summary,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()