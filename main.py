import argparse
import copy
import csv
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model import Model
from role import InteractionRoleEngine
from train_baselines import (
    compute_class_weights,
    evaluate,
    get_loader,
    load_external_metrics,
    load_matrix,
    log_message,
    save_external_metrics,
    save_matrix,
    summarize_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="XLS-R + interaction-aware continual learning under the baseline protocol"
    )

    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--ssl_checkpoint", type=str, default="/root/autodl-fs/xlsr2_300m.pt")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--engine_path",
        type=str,
        default=None,
        help="Optional checkpoint with saved InteractionRoleEngine state.",
    )

    parser.add_argument("--start_task_idx", type=int, default=0)
    parser.add_argument("--train_tasks", type=str, default="ASVspoof2019_LA,for-norm,XMAD,ASVspoof5")
    parser.add_argument(
        "--external_tasks",
        type=str,
        default="release_in_the_wild,CodecFake,ASVspoof2021_LA_eval,ASVspoof2021_DF_eval,MLAAD_M-AILABS",
    )
    parser.add_argument("--tasks", type=str, default=None, help="Deprecated alias for --train_tasks.")

    parser.add_argument("--output_dir", type=str, default="runs/irnr")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--eval_batch_size", type=int, default=14)
    parser.add_argument("--num_epochs_per_task", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=1)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)

    parser.add_argument("--balanced_ce", action="store_true", default=True)
    parser.add_argument("--disable_balanced_ce", action="store_false", dest="balanced_ce")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--freeze_ssl", action="store_true", default=True)
    parser.add_argument("--unfreeze_ssl", action="store_false", dest="freeze_ssl")

    parser.add_argument("--mem_momentum", type=float, default=0.95)
    parser.add_argument("--grad_momentum", type=float, default=0.9)
    parser.add_argument("--role_activation_tau", type=float, default=1.0)
    parser.add_argument("--role_temperature", type=float, default=0.7)
    parser.add_argument("--alpha_irnr", type=float, default=0.3)
    parser.add_argument("--lambda_delta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--drift_warmup_epochs",
        type=int,
        default=5
    )

    # parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--penalty_rho", type=float, default=0.05)
    parser.add_argument("--lambda_lr", type=float, default=0.05)
    parser.add_argument("--eps0", type=float, default=1e-3)
    parser.add_argument("--lambda_max", type=float, default=10)
    parser.add_argument("--eta_bona", type=float, default=0.7)
    parser.add_argument("--eta_spoof", type=float, default=0.3)
    parser.add_argument("--delta_max", type=float, default=3.0)
    parser.add_argument("--drift_scale_floor", type=float, default=1e-6)
    parser.add_argument("--delta_abs_tau", type=float, default=0.02)
    parser.add_argument("--lagrange_task_decay", type=float, default=0.0)
    parser.add_argument("--spoof_parallel_ratio", type=float, default=0.2)
    parser.add_argument("--conflict_scale_shared", type=float, default=0.0)
    parser.add_argument("--conflict_scale_bona", type=float, default=0.1)
    parser.add_argument("--conflict_scale_spoof", type=float, default=0.2)
    parser.add_argument("--conflict_scale_free", type=float, default=0.5)
    parser.add_argument("--irnr_diagnostics", action="store_true")
    parser.add_argument("--irnr_diagnostics_interval", type=int, default=1)
    parser.add_argument(
        "--save_mechanism_data",
        action="store_true",
        help=(
            "Save per-step gradient components and per-neuron role/drift "
            "snapshots for mechanism analysis."
        ),
    )
    parser.add_argument(
        "--mechanism_data_interval",
        type=int,
        default=100,
        help="Record detailed gradient components every N training steps.",
    )
    parser.add_argument(
        "--mechanism_data_dir",
        type=str,
        default=None,
        help="Optional mechanism-data directory; defaults to RUN_DIR/metrics/mechanism.",
    )

    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation and only train/log reg_loss.")

    parser.add_argument(
        "--skip_external_eval",
        action="store_true",
        help="Skip external dataset evaluation."
    )

    parser.add_argument(
        "--projection_margin",
        type=float,
        default=1.5,
        help="Hard projection boundary multiplier over the soft drift budget.",
    )

    parser.add_argument(
        "--distill_weight",
        type=float,
        default=1.0,
        help="Weight of LwF-style knowledge distillation loss."
    )

    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=2.0,
        help="Temperature for knowledge distillation."
    )

    return parser.parse_args()


def build_model(args, device):
    model = Model(args, device).to(device)

    checkpoint_path = args.model_path or args.engine_path
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        cleaned = OrderedDict()
        for key, value in state_dict.items():
            cleaned[key[7:] if key.startswith("module.") else key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[*] Loaded model checkpoint={checkpoint_path}")
        print(f"[*] Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    if args.freeze_ssl:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
        print("[*] Frozen XLS-R backbone parameters")
    else:
        print("[*] XLS-R backbone is trainable")

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"[*] Trainable parameters: {trainable} / {total}")

    return model


def build_engine(model, args):
    return InteractionRoleEngine(
        model,
        mem_momentum=args.mem_momentum,
        grad_momentum=args.grad_momentum,
        role_activation_tau=args.role_activation_tau,
        role_temperature=args.role_temperature,
        alpha_irnr=args.alpha_irnr,
        lambda_delta=args.lambda_delta,
        # rho=args.rho,
        penalty_rho=args.penalty_rho,
        lambda_lr=args.lambda_lr,
        eps0=args.eps0,
        lambda_max=args.lambda_max,
        eta_bona=args.eta_bona,
        eta_spoof=args.eta_spoof,
        delta_max=args.delta_max,
        drift_scale_floor=args.drift_scale_floor,
        delta_abs_tau=args.delta_abs_tau,
        lagrange_task_decay=args.lagrange_task_decay,
        spoof_parallel_ratio=args.spoof_parallel_ratio,
        conflict_scale_shared=args.conflict_scale_shared,
        conflict_scale_bona=args.conflict_scale_bona,
        conflict_scale_spoof=args.conflict_scale_spoof,
        conflict_scale_free=args.conflict_scale_free,
        projection_margin=args.projection_margin,
        diagnostics_enabled=args.irnr_diagnostics,
        mechanism_data_enabled=args.save_mechanism_data,
        mechanism_data_interval=args.mechanism_data_interval,
    )



def _trainable_params(model):
    """Return trainable parameters in a stable order."""
    return [param for param in model.parameters() if param.requires_grad]


def _autograd_grads(loss, params, retain_graph):
    """
    Compute gradients without accumulating into .grad.
    Returns a list aligned with params. If loss has no gradient graph,
    return all None.
    """
    if loss is None or not loss.requires_grad:
        return [None] * len(params)

    return torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _write_grads(params, grads):
    """Overwrite parameter .grad with the supplied gradient list."""
    for param, grad in zip(params, grads):
        if grad is None:
            param.grad = None
        else:
            param.grad = grad.detach().clone()


def _clone_current_grads(params):
    """Clone the current .grad values, preserving None entries."""
    return [
        None if param.grad is None else param.grad.detach().clone()
        for param in params
    ]


def _combine_grads(*grad_groups):
    """
    Element-wise sum of multiple gradient groups.
    Each group must be aligned with the same parameter list.
    """
    if not grad_groups:
        return []

    combined = []
    for grads_at_param in zip(*grad_groups):
        total = None
        for grad in grads_at_param:
            if grad is None:
                continue
            total = grad.detach().clone() if total is None else total + grad.detach()

        combined.append(total)

    return combined


def append_dict_rows(path, rows):
    """Append homogeneous dictionaries to CSV and create the header once."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_dict_rows(path, rows):
    """Atomically replace one task-level mechanism snapshot CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    fieldnames = list(rows[0].keys())
    with open(temporary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def safe_filename(value):
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value
    )

def train_one_epoch(
    task_name,
    task_idx,
    loader,
    engine,
    optimizer,
    criterion,
    device,
    epoch,
    gamma,
    teacher_classifier=None,
    teacher_feature_cache=None,
    distill_weight=1.0,
    distill_temperature=2.0,
):
    engine.model.train()
    if teacher_classifier is not None:
        teacher_classifier.eval()

    running_loss = 0.0
    running_reg = 0.0
    running_kd = 0.0
    running_kd_over_cls = 0.0
    num_correct = 0
    num_total = 0

    pbar = tqdm(loader, desc=f"irnr {task_name} epoch {epoch}", leave=True)

    # Clear epoch-level violation statistics.
    engine.reset_epoch_violations()

    for step, (batch_x, batch_y) in enumerate(pbar):
        engine.set_diagnostic_context(task_idx, task_name, epoch, step)
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).long().to(device)

        optimizer.zero_grad(set_to_none=True)
        engine.before_forward(batch_y)

        # Student forward. The classifier forward-pre-hook installed at the
        # beginning of the task captures the pooled feature entering the
        # current classifier, so XLS-R is executed only once per batch.
        if teacher_feature_cache is not None:
            teacher_feature_cache.clear()

        logits = engine.model(batch_x)
        loss_cls = criterion(logits, batch_y)

        # LwF-style functional distillation with a classifier-only teacher.
        loss_kd = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if teacher_classifier is not None:
            if teacher_feature_cache is None or "features" not in teacher_feature_cache:
                raise RuntimeError(
                    "Classifier input feature was not captured. "
                    "Check that engine.model.classifier is called inside Model.forward()."
                )

            shared_features = teacher_feature_cache["features"]

            with torch.no_grad():
                teacher_logits = teacher_classifier(shared_features)

            temperature = distill_temperature
            loss_kd = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)

        # Freeze the current batch's role assignment before drift regularization.
        engine.prepare_role_snapshot()
        loss_reg = engine.compute_lagrangian_reg()

        # --------------------------------------------------------------
        # V12: gradient-decoupled optimization with a local KD reference
        #
        #   g_final =
        #       ConflictOnlyRoleWarp(g_cls; ref=g_kd)
        #       + distill_weight * g_kd
        #       + gamma * g_reg
        #
        # Role estimation and RoleWarp therefore act only on the
        # classification objective. KD and drift gradients remain
        # independent retention signals and are not warped a second time.
        # --------------------------------------------------------------

        params = _trainable_params(engine.model)

        # 1) LwF/KD gradient. Besides being added unchanged below, this is the
        # local old-knowledge direction used to identify a genuine CE conflict
        # at the current parameter point.
        # KD is a retention signal, not class-conditional evidence for the
        # role estimator used by the next batch.
        engine.collect_role_memory = False
        try:
            kd_grads = _autograd_grads(
                distill_weight * loss_kd,
                params,
                retain_graph=True,
            )
        finally:
            engine.collect_role_memory = True

        kd_reference_grads = {
            id(param): grad
            for param, grad in zip(params, kd_grads)
            if grad is not None
        }

        # 2) Classification gradient. This is the only backward pass allowed
        # to update current-task role memory.
        cls_grads = _autograd_grads(
            loss_cls,
            params,
            retain_graph=False,
        )

        # InteractionRoleEngine operates on .grad, so temporarily write only
        # CE gradients and gate their KD-conflicting component.
        _write_grads(params, cls_grads)
        engine.warp_gradients(reference_grads=kd_reference_grads)
        warped_cls_grads = _clone_current_grads(params)

        # 3) Drift gradient -- already role-weighted inside the regularizer,
        # so it also bypasses RoleWarp.
        reg_grads = _autograd_grads(
            gamma * loss_reg,
            params,
            retain_graph=False,
        )

        # 4) Final optimizer gradient.
        final_grads = _combine_grads(
            warped_cls_grads,
            kd_grads,
            reg_grads,
        )
        _write_grads(params, final_grads)

        optimizer.step()

        # Hard projection remains only an outer safety boundary.
        engine.project_drift_constraint()

        engine.after_step()
        engine.record_loss_diagnostics(loss_cls, loss_reg)

        kd_over_cls = (
            distill_weight * loss_kd.detach()
            / (loss_cls.detach() + 1e-12)
        )

        running_loss += loss_cls.item() * batch_size
        running_reg += loss_reg.item() * batch_size
        running_kd += loss_kd.item() * batch_size
        running_kd_over_cls += kd_over_cls.item() * batch_size
        num_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        num_total += batch_size

        pbar.set_postfix(
            loss=f"{loss_cls.item():.4f}",
            kd=f"{loss_kd.item():.4f}",
            reg=f"{loss_reg.item():.4f}",
            acc=f"{100.0 * num_correct / max(num_total, 1):.2f}%",
        )

    return (
        running_loss / max(num_total, 1),
        running_reg / max(num_total, 1),
        running_kd / max(num_total, 1),
        running_kd_over_cls / max(num_total, 1),
        100.0 * num_correct / max(num_total, 1),
    )

def write_config(args, run_dir, log_dir):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        config = vars(args).copy()
        config["resolved_run_dir"] = run_dir
        config["resolved_log_dir"] = log_dir
        json.dump(config, f, indent=2)


def ensure_train_epoch_csv(path):
    header = [
        "timestamp",
        "task_idx",
        "task",
        "epoch",
        "loss",
        "reg_loss",
        "kd_loss",
        "kd_over_cls",
        "acc",
        "best_train_loss",
        "weight_real",
        "weight_fake",
        "bad_epochs",
        "early_stopped",
        "gamma",
        "lambda_delta",
        # "rho",
        "penalty_rho",
        "lambda_lr",
        "eps0",
        "lambda_max",
        "eta_bona",
        "eta_spoof",
        "delta_max",
        "drift_scale_floor",
        "delta_abs_tau",
        "lagrange_task_decay",
        "spoof_parallel_ratio",
        "conflict_scale_shared",
        "conflict_scale_bona",
        "conflict_scale_spoof",
        "conflict_scale_free",
        "irnr_diagnostics",
        "irnr_diagnostics_interval",
        "mem_momentum",
        "grad_momentum",
        "role_activation_tau",
        "role_temperature",
        "alpha_irnr",
        "distill_weight",
        "distill_temperature",
        "projection_margin",
        "lr",
        "weight_decay",
        "irnr_diagnostics_json",
    ]

    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    old_header = rows[0]
    changed = False

    if "delta_kappa" in old_header:
        remove_idx = old_header.index("delta_kappa")
        old_header.pop(remove_idx)
        for row in rows[1:]:
            if len(row) > remove_idx:
                row.pop(remove_idx)
        changed = True

    for column in (
        "delta_max",
        "drift_scale_floor",
        "delta_abs_tau",
        "lagrange_task_decay",
        "spoof_parallel_ratio",
        "conflict_scale_shared",
        "conflict_scale_bona",
        "conflict_scale_spoof",
        "conflict_scale_free",
        "irnr_diagnostics",
        "irnr_diagnostics_interval",
        "alpha_irnr",
        "role_activation_tau",
        "irnr_diagnostics_json",
        "kd_loss",
        "kd_over_cls",
        "distill_weight",
        "distill_temperature",
        "projection_margin",
    ):
        if column not in old_header:
            insert_idx = header.index(column)
            old_header.insert(insert_idx, column)
            for row in rows[1:]:
                row.insert(insert_idx, "")
            changed = True

    if not changed:
        return

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def append_train_epoch(
    path,
    args,
    task_idx,
    task_name,
    epoch,
    loss,
    reg_loss,
    kd_loss,
    kd_over_cls,
    acc,
    best_loss,
    weights,
    bad_epochs,
    early_stopped,
    diagnostics_json,
):
    weight_real = float(weights[0].item()) if weights is not None else ""
    weight_fake = float(weights[1].item()) if weights is not None else ""

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            task_idx,
            task_name,
            epoch,
            f"{loss:.8f}",
            f"{reg_loss:.8f}",
            f"{kd_loss:.8f}",
            f"{kd_over_cls:.8f}",
            f"{acc:.6f}",
            f"{best_loss:.8f}",
            weight_real,
            weight_fake,
            bad_epochs,
            int(early_stopped),
            args.gamma,
            args.lambda_delta,
            # args.rho,
            args.penalty_rho,
            args.lambda_lr,
            args.eps0,
            args.lambda_max,
            args.eta_bona,
            args.eta_spoof,
            args.delta_max,
            args.drift_scale_floor,
            args.delta_abs_tau,
            args.lagrange_task_decay,
            args.spoof_parallel_ratio,
            args.conflict_scale_shared,
            args.conflict_scale_bona,
            args.conflict_scale_spoof,
            args.conflict_scale_free,
            int(args.irnr_diagnostics),
            args.irnr_diagnostics_interval,
            args.mem_momentum,
            args.grad_momentum,
            args.role_activation_tau,
            args.role_temperature,
            args.alpha_irnr,
            args.distill_weight,
            args.distill_temperature,
            args.projection_margin,
            args.lr,
            args.weight_decay,
            diagnostics_json,
        ])


def main():
    args = parse_args()
    if args.irnr_diagnostics_interval <= 0:
        raise ValueError("irnr_diagnostics_interval must be > 0")
    if args.mechanism_data_interval <= 0:
        raise ValueError("mechanism_data_interval must be > 0")
    if args.distill_temperature <= 0:
        raise ValueError("distill_temperature must be > 0")
    if args.distill_weight < 0:
        raise ValueError("distill_weight must be >= 0")
    if args.projection_margin < 1.0:
        raise ValueError("projection_margin must be >= 1.0")
    if args.distill_weight > 0 and not args.freeze_ssl:
        raise ValueError(
            "Classifier-only LwF teacher requires --freeze_ssl because "
            "teacher and student must share the same frozen XLS-R features."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tasks is not None:
        args.train_tasks = args.tasks

    train_tasks = [task.strip() for task in args.train_tasks.split(",") if task.strip()]
    external_tasks = [task.strip() for task in args.external_tasks.split(",") if task.strip()]

    run_dir = args.run_dir if args.run_dir is not None else os.path.join(args.output_dir, f"seed_{args.seed}")
    log_dir = args.log_dir if args.log_dir is not None else os.path.join(run_dir, "logs")
    mechanism_data_dir = (
        args.mechanism_data_dir
        if args.mechanism_data_dir is not None
        else os.path.join(run_dir, "metrics", "mechanism")
    )
    args.resolved_mechanism_data_dir = mechanism_data_dir

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "scores"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if args.save_mechanism_data:
        os.makedirs(mechanism_data_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "train.log")
    train_epoch_path = os.path.join(run_dir, "metrics", "train_epochs.csv")
    external_metrics_path = os.path.join(run_dir, "metrics", "external_metrics.csv")

    write_config(args, run_dir, log_dir)
    ensure_train_epoch_csv(train_epoch_path)

    log_message(log_file, f"[*] Run directory: {run_dir}")
    log_message(log_file, f"[*] Log directory: {log_dir}")
    log_message(log_file, f"[*] Device: {device}")
    log_message(log_file, "[*] Method: irnr_v12_cross_task_kd_ref")
    log_message(log_file, "[*] Gradient rule: ConflictOnlyRoleWarp(g_cls; ref=g_kd) + beta*g_kd + gamma*g_reg")
    log_message(log_file, f"[*] Train tasks: {train_tasks}")
    log_message(log_file, f"[*] External tasks: {external_tasks}")
    if args.save_mechanism_data:
        log_message(
            log_file,
            f"[*] Mechanism data: {mechanism_data_dir} "
            f"(gradient interval={args.mechanism_data_interval})",
        )

    eer_matrix = load_matrix(os.path.join(run_dir, "metrics", "eer_matrix.csv"))
    auc_matrix = load_matrix(os.path.join(run_dir, "metrics", "auc_matrix.csv"))
    external_rows = load_external_metrics(external_metrics_path)

    if args.start_task_idx > 0 and args.engine_path is None and args.model_path is None:
        raise ValueError(
            "--start_task_idx > 0 requires --model_path or --engine_path, "
            "otherwise IRNR anchor/memory state is missing."
        )
    if args.start_task_idx > 0 and args.engine_path is None:
        log_message(
            log_file,
            "[warning] --start_task_idx was set without --engine_path; "
            "model weights may resume, but IRNR anchor/memory state will be incomplete.",
        )

    model = build_model(args, device)
    engine = build_engine(model, args)
    engine_checkpoint_path = args.engine_path or args.model_path
    if engine_checkpoint_path:
        checkpoint = torch.load(engine_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "engine" in checkpoint:
            engine.load_state(checkpoint["engine"])
            log_message(log_file, f"[*] Loaded engine state from: {engine_checkpoint_path}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, engine.model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer_checkpoint_path = args.engine_path or args.model_path
    if optimizer_checkpoint_path:
        checkpoint = torch.load(optimizer_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            log_message(log_file, f"[*] Loaded optimizer state from: {optimizer_checkpoint_path}")

    for task_idx, task_name in enumerate(train_tasks):
        if task_idx < args.start_task_idx:
            log_message(log_file, f"[*] Skip completed task={task_name} idx={task_idx}")
            continue

        # Classifier-only LwF teacher:
        # XLS-R is frozen, so the teacher and student share exactly the same
        # backbone feature. We only snapshot the previous best classifier.
        teacher_classifier = None
        teacher_feature_cache = None
        teacher_feature_hook = None

        if task_idx > 0 and args.distill_weight > 0:
            if not hasattr(engine.model, "classifier"):
                raise AttributeError(
                    "Classifier-only LwF requires engine.model.classifier."
                )

            teacher_classifier = copy.deepcopy(engine.model.classifier).to(device)
            # deepcopy() also copies the student's IRNR forward hooks. A
            # teacher forward would otherwise overwrite engine.cached_outputs
            # and corrupt the subsequent CE role-memory update.
            engine.strip_runtime_hooks(teacher_classifier)
            teacher_classifier.eval()
            for param in teacher_classifier.parameters():
                param.requires_grad_(False)

            # Capture the input of the *student* classifier during the normal
            # model forward. This preserves all existing IRNR hooks on
            # classifier.0 / classifier.3 and avoids a second XLS-R forward.
            teacher_feature_cache = {}

            def _capture_classifier_input(module, inputs):
                if not inputs:
                    raise RuntimeError("Classifier forward_pre_hook received no input.")
                teacher_feature_cache["features"] = inputs[0].detach()

            teacher_feature_hook = engine.model.classifier.register_forward_pre_hook(
                _capture_classifier_input
            )

            log_message(
                log_file,
                f"[*] Built frozen classifier-only LwF teacher before "
                f"task={task_name} idx={task_idx}",
            )

        engine.begin_task()

        class_weights = compute_class_weights(task_name, args, device) if args.balanced_ce else None
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        weight_real = float(class_weights[0].item()) if class_weights is not None else ""
        weight_fake = float(class_weights[1].item()) if class_weights is not None else ""

        log_message(
            log_file,
            f"[*] Start task={task_name} idx={task_idx} "
            f"weight_real={weight_real} weight_fake={weight_fake}",
        )

        train_loader = get_loader(task_name, args, "train", shuffle=True, drop_last=True)
        best_train_loss = float("inf")
        best_model_state = None
        best_engine_state = None
        best_optimizer_state = None
        best_neuron_snapshot = []
        best_epoch = None
        bad_epochs = 0
        mechanism_task_tag = f"task_{task_idx + 1}_{safe_filename(task_name)}"
        gradient_components_path = os.path.join(
            mechanism_data_dir,
            f"gradient_components_{mechanism_task_tag}.csv",
        )

        for epoch in range(args.num_epochs_per_task):
            # loss, reg_loss, acc = train_one_epoch(
            #     task_name=task_name,
            #     task_idx=task_idx,
            #     loader=train_loader,
            #     engine=engine,
            #     optimizer=optimizer,
            #     criterion=criterion,
            #     device=device,
            #     epoch=epoch,
            #     gamma=args.gamma,
            # )
            # ===== drift warmup =====
            if args.drift_warmup_epochs > 0:
                gamma_now = args.gamma * min(
                    1.0,
                    (epoch + 1) / args.drift_warmup_epochs
                )
            else:
                gamma_now = args.gamma

            loss, reg_loss, kd_loss, kd_over_cls, acc = train_one_epoch(
                task_name=task_name,
                task_idx=task_idx,
                loader=train_loader,
                engine=engine,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                gamma=gamma_now,
                teacher_classifier=teacher_classifier,
                teacher_feature_cache=teacher_feature_cache,
                distill_weight=args.distill_weight,
                distill_temperature=args.distill_temperature,
            )

            # ===== epoch结束更新lambda =====
            engine.update_lagrange_multipliers_epoch()

            # log_message(
            #     log_file,
            #     f"[train] task={task_name} epoch={epoch} "
            #     f"loss={loss:.4f} reg_loss={reg_loss:.6f} acc={acc:.2f}",
            # )
            log_message(
                log_file,
                f"[train] task={task_name} epoch={epoch} "
                f"gamma={gamma_now:.4f} "
                f"loss={loss:.4f} "
                f"reg_loss={reg_loss:.6f} "
                f"kd_loss={kd_loss:.6f} "
                f"kd_over_cls={kd_over_cls:.6f} "
                f"acc={acc:.2f}",
            )

            diagnostics_json = ""
            if args.irnr_diagnostics and epoch % args.irnr_diagnostics_interval == 0:
                diagnostics = engine.get_diagnostics_summary(reset=True)
                diagnostics_json = json.dumps(diagnostics, sort_keys=True)
                log_message(log_file, f"[irnr-diagnostics] task={task_name} epoch={epoch} {diagnostics_json}")
            elif args.irnr_diagnostics:
                engine.get_diagnostics_summary(reset=True)

            if args.save_mechanism_data:
                append_dict_rows(
                    gradient_components_path,
                    engine.pop_gradient_component_records(),
                )

            early_stopped = False
            if loss < best_train_loss - args.early_stop_min_delta:
                best_train_loss = loss
                best_model_state = copy.deepcopy(engine.model.state_dict())
                best_engine_state = engine.get_state()
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_epoch = epoch
                if args.save_mechanism_data:
                    best_neuron_snapshot = engine.get_neuron_diagnostics_snapshot()
                bad_epochs = 0
            else:
                bad_epochs += 1
                log_message(
                    log_file,
                    f"[early-stop] task={task_name} epoch={epoch} "
                    f"train_loss did not improve. bad_epochs={bad_epochs}/{args.early_stop_patience}",
                )

                if bad_epochs >= args.early_stop_patience:
                    early_stopped = True
                    log_message(log_file, f"[early-stop] stop task={task_name} at epoch={epoch}")

            append_train_epoch(
                train_epoch_path,
                args,
                task_idx,
                task_name,
                epoch,
                loss,
                reg_loss,
                kd_loss,
                kd_over_cls,
                acc,
                best_train_loss,
                class_weights,
                bad_epochs,
                early_stopped,
                diagnostics_json,
            )

            if early_stopped:
                break

        if best_model_state is not None:
            engine.model.load_state_dict(best_model_state)
            engine.load_state(best_engine_state)
            optimizer.load_state_dict(best_optimizer_state)
            log_message(log_file, f"[*] Restored best model for task={task_name}, train_loss={best_train_loss:.4f}")

        if args.save_mechanism_data:
            snapshot_rows = [
                {
                    "task_idx": task_idx,
                    "task": task_name,
                    "epoch": best_epoch,
                    **row,
                }
                for row in best_neuron_snapshot
            ]
            snapshot_path = os.path.join(
                mechanism_data_dir,
                f"role_importance_drift_{mechanism_task_tag}.csv",
            )
            if snapshot_rows:
                write_dict_rows(snapshot_path, snapshot_rows)
                log_message(
                    log_file,
                    f"[*] Saved mechanism snapshot rows={len(snapshot_rows)} "
                    f"to: {snapshot_path}",
                )
            else:
                log_message(
                    log_file,
                    f"[*] No cross-task mechanism snapshot for task={task_name}; "
                    "a historical task anchor is required.",
                )

        log_message(log_file, f"[*] Updating anchor after task={task_name}")
        engine.update_task_anchor()

        checkpoint_path = os.path.join(run_dir, "checkpoints", f"task_{task_idx + 1}_{task_name}_best.pt")
        torch.save(
            {
                "task_idx": task_idx,
                "task_name": task_name,
                "best_train_loss": best_train_loss,
                "model": engine.model.state_dict(),
                "engine": engine.get_state(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )
        log_message(log_file, f"[*] Saved best checkpoint to: {checkpoint_path}")

        # if args.skip_eval:
        #     log_message(log_file, f"[*] Skip evaluation after task={task_name}")
        #     save_matrix(os.path.join(run_dir, "metrics", "eer_matrix.csv"), train_tasks, train_tasks, eer_matrix)
        #     save_matrix(os.path.join(run_dir, "metrics", "auc_matrix.csv"), train_tasks, train_tasks, auc_matrix)
        #     save_external_metrics(external_metrics_path, external_rows)
        #
        #     del train_loader
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #
        #     continue
        if task_idx == 0:
            log_message(
                log_file,
                f"[*] Skip first task evaluation task={task_name}"
            )

            save_matrix(
                os.path.join(run_dir, "metrics", "eer_matrix.csv"),
                train_tasks,
                train_tasks,
                eer_matrix
            )

            save_matrix(
                os.path.join(run_dir, "metrics", "auc_matrix.csv"),
                train_tasks,
                train_tasks,
                auc_matrix
            )

            save_external_metrics(
                external_metrics_path,
                external_rows
            )

            del train_loader
            if teacher_feature_hook is not None:
                teacher_feature_hook.remove()
            if teacher_classifier is not None:
                del teacher_classifier
            teacher_feature_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        if args.skip_eval:
            log_message(
                log_file,
                f"[*] Skip evaluation after task={task_name}"
            )

            save_matrix(
                os.path.join(run_dir, "metrics", "eer_matrix.csv"),
                train_tasks,
                train_tasks,
                eer_matrix
            )

            save_matrix(
                os.path.join(run_dir, "metrics", "auc_matrix.csv"),
                train_tasks,
                train_tasks,
                auc_matrix
            )

            save_external_metrics(
                external_metrics_path,
                external_rows
            )

            del train_loader
            if teacher_feature_hook is not None:
                teacher_feature_hook.remove()
            if teacher_classifier is not None:
                del teacher_classifier
            teacher_feature_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        eer_matrix[task_name] = {}
        auc_matrix[task_name] = {}

        for eval_task in train_tasks[: task_idx + 1]:
            eval_loader = get_loader(eval_task, args, "test", shuffle=False, drop_last=False, return_path=True)
            score_path = os.path.join(
                run_dir,
                "scores",
                f"after_{task_idx + 1}_{task_name}__test_{eval_task}.csv",
            )
            score_txt_path = os.path.join(
                run_dir,
                "scores",
                f"after_{task_idx + 1}_{task_name}__test_{eval_task}_scores.txt",
            )

            metrics = evaluate(
                engine.model,
                eval_loader,
                device,
                score_path=score_path,
                score_txt_path=score_txt_path,
            )
            eer_matrix[task_name][eval_task] = metrics["eer"]
            auc_matrix[task_name][eval_task] = metrics["auc"]

            log_message(
                log_file,
                f"[eval] after={task_name} test={eval_task} "
                f"EER={metrics['eer']:.6f} AUC={metrics['auc']:.6f} scores={score_txt_path}",
            )

        if not args.skip_external_eval:
            for external_task in external_tasks:
                eval_loader = get_loader(external_task, args, "test", shuffle=False, drop_last=False, return_path=True)
                score_path = os.path.join(
                    run_dir,
                    "scores",
                    f"after_{task_idx + 1}_{task_name}__external_{external_task}.csv",
                )
                score_txt_path = os.path.join(
                    run_dir,
                    "scores",
                    f"after_{task_idx + 1}_{task_name}__external_{external_task}_scores.txt",
                )

                metrics = evaluate(
                    engine.model,
                    eval_loader,
                    device,
                    score_path=score_path,
                    score_txt_path=score_txt_path,
                )

                external_rows.append({
                    "train_step": task_name,
                    "external_task": external_task,
                    "eer": metrics["eer"],
                    "auc": metrics["auc"],
                    "score_path": score_txt_path,
                })

                log_message(
                    log_file,
                    f"[external] after={task_name} test={external_task} "
                    f"EER={metrics['eer']:.6f} AUC={metrics['auc']:.6f} scores={score_txt_path}",
                )

        save_matrix(os.path.join(run_dir, "metrics", "eer_matrix.csv"), train_tasks, train_tasks, eer_matrix)
        save_matrix(os.path.join(run_dir, "metrics", "auc_matrix.csv"), train_tasks, train_tasks, auc_matrix)
        save_external_metrics(external_metrics_path, external_rows)

        del train_loader
        if teacher_feature_hook is not None:
            teacher_feature_hook.remove()
        if teacher_classifier is not None:
            del teacher_classifier
        teacher_feature_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # summary = summarize_metrics(train_tasks, eer_matrix, auc_matrix)

    if args.skip_eval:
        summary = {
            "final_avg_eer": float("nan"),
            "final_avg_auc": float("nan"),
            "af_eer": float("nan"),
            "af_auc": float("nan"),
            "external_final_avg_eer": float("nan"),
            "external_final_avg_auc": float("nan"),
        }
    else:
        summary = summarize_metrics(train_tasks, eer_matrix, auc_matrix)

        if external_rows:
            final_external = [row for row in external_rows if row["train_step"] == train_tasks[-1]]
            if final_external:
                summary["external_final_avg_eer"] = float(np.nanmean([float(row["eer"]) for row in final_external]))
                summary["external_final_avg_auc"] = float(np.nanmean([float(row["auc"]) for row in final_external]))
            else:
                summary["external_final_avg_eer"] = float("nan")
                summary["external_final_avg_auc"] = float("nan")

    if external_rows:
        final_external = [row for row in external_rows if row["train_step"] == train_tasks[-1]]
        if final_external:
            summary["external_final_avg_eer"] = float(np.nanmean([float(row["eer"]) for row in final_external]))
            summary["external_final_avg_auc"] = float(np.nanmean([float(row["auc"]) for row in final_external]))
        else:
            summary["external_final_avg_eer"] = float("nan")
            summary["external_final_avg_auc"] = float("nan")

    with open(os.path.join(run_dir, "metrics", "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log_message(log_file, json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
