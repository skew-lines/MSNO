import copy

import torch
import torch.nn as nn


class InteractionRoleEngine:
    """
    Cross-task IRNR.
    - frozen historical importance versus current-task importance
    - conflict-only residual gradient modulation
    - task-balanced historical saliency and EMA task memories
    - legacy checkpoint and training-loop compatibility
    """

    def __init__(
        self,
        model: nn.Module,
        mem_momentum: float = 0.9,
        grad_momentum: float = 0.9,
        role_activation_tau: float = 1.0,
        role_temperature: float = 0.7,
        lambda_delta: float = 0.1,
        # rho: float = 0.05,
        penalty_rho=0.05,
        lambda_lr = 0.005,
        eps0: float = 1e-3,
        lambda_max: float = 100.0,
        eta_bona: float = 0.7,
        eta_spoof: float = 0.3,
        delta_max: float = 3.0,
        drift_scale_floor: float = 1e-6,
        delta_abs_tau: float = 0.02,
        lagrange_task_decay: float = 0.0,
        spoof_parallel_ratio: float = 0.2,
        conflict_scale_shared: float = 0.0,
        conflict_scale_bona: float = 0.1,
        conflict_scale_spoof: float = 0.2,
        conflict_scale_free: float = 0.5,
        eps: float = 1e-8,
        alpha_irnr: float = 0.3,
        projection_margin: float = 1.2,
        diagnostics_enabled: bool = False,
        mechanism_data_enabled: bool = False,
        mechanism_data_interval: int = 100,
    ):
        if projection_margin < 1.0:
            raise ValueError("projection_margin must be >= 1.0")
        if mechanism_data_interval <= 0:
            raise ValueError("mechanism_data_interval must be > 0")

        if drift_scale_floor <= 0:
            raise ValueError("drift_scale_floor must be > 0")
        if delta_abs_tau <= 0:
            raise ValueError("delta_abs_tau must be > 0")
        if role_activation_tau <= 0:
            raise ValueError("role_activation_tau must be > 0")
        if role_temperature <= 0:
            raise ValueError("role_temperature must be > 0")
        if not 0.0 <= lagrange_task_decay <= 1.0:
            raise ValueError("lagrange_task_decay must be in [0, 1]")
        for name, value in {
            "spoof_parallel_ratio": spoof_parallel_ratio,
            "conflict_scale_shared": conflict_scale_shared,
            "conflict_scale_bona": conflict_scale_bona,
            "conflict_scale_spoof": conflict_scale_spoof,
            "conflict_scale_free": conflict_scale_free,
            "alpha_irnr": alpha_irnr,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self.model = model

        self.mem_momentum = mem_momentum
        self.grad_momentum = grad_momentum
        self.role_activation_tau = role_activation_tau
        self.role_temperature = role_temperature

        self.lambda_delta = lambda_delta
        # self.rho = rho
        self.penalty_rho = penalty_rho
        self.lambda_lr = lambda_lr
        self.eps0 = eps0
        self.lambda_max = lambda_max
        self.eta_bona = eta_bona
        self.eta_spoof = eta_spoof
        self.delta_max = delta_max
        self.drift_scale_floor = drift_scale_floor
        self.delta_abs_tau = delta_abs_tau
        self.lagrange_task_decay = lagrange_task_decay
        self.spoof_parallel_ratio = spoof_parallel_ratio
        self.conflict_scale_shared = conflict_scale_shared
        self.conflict_scale_bona = conflict_scale_bona
        self.conflict_scale_old = conflict_scale_bona  # semantic alias
        self.conflict_scale_spoof = conflict_scale_spoof
        self.conflict_scale_free = conflict_scale_free
        self.eps = eps
        self.alpha_irnr = alpha_irnr
        self.diagnostics_enabled = diagnostics_enabled
        self.mechanism_data_enabled = mechanism_data_enabled
        self.mechanism_data_interval = mechanism_data_interval

        self.global_mem_real = {}
        self.global_mem_fake = {}
        self.num_consolidated_tasks = 0
        self.task_mem_real = {}
        self.task_mem_fake = {}
        self.grad_ref = {}
        self.grad_running = {}
        self.prev_role_probs = {}
        self.lagrange_multipliers = {}
        self.last_violations = {}
        self.diagnostics = {}
        self.current_labels = None
        self.collect_role_memory = True
        self.current_role_snapshot = {}
        # Frozen at begin_task(): statistics accumulated before the current
        # task. Retention must not be inferred from current-task saliency.
        self.historical_role_snapshot = {}
        self.module_map = {}
        self.cached_outputs = {}
        self.anchor_weights = {}
        self.diagnostic_context = {}
        self.gradient_component_records = []
        self.task_projection_hits = {}
        self.task_projection_checks = {}
        self.task_projection_scale_sum = {}
        self.task_projection_scale_min = {}

        self._setup_hooks()

        self.epoch_violations = {}

        self.projection_margin = projection_margin

    # -------------------------
    # hooks
    # -------------------------
    def _setup_hooks(self):
        for name, module in self.model.named_modules():
            if "ssl_model" in name:
                continue

            if isinstance(module, nn.Linear):
                self.module_map[name] = module
                module.register_forward_hook(self._forward_hook(name))

    @staticmethod
    def strip_runtime_hooks(module):
        """Remove hooks copied from the student before using a teacher clone."""
        for child in module.modules():
            child._forward_hooks.clear()
            child._forward_pre_hooks.clear()
            child._backward_hooks.clear()

    def _forward_hook(self, name):
        def hook(module, inp, out):
            self.cached_outputs[name] = out
            if out.requires_grad:
                out.register_hook(self._activation_grad_hook(name))
        return hook

    def _activation_grad_hook(self, name):
        def hook(grad_output):
            if not self.collect_role_memory or self.current_labels is None or name not in self.cached_outputs:
                return grad_output

            output = self.cached_outputs[name].detach()
            grad = grad_output.detach()
            labels = self.current_labels.detach()

            if output.dim() != 2 or output.shape[0] != labels.shape[0]:
                return grad_output

            contribution = (output * grad).abs()
            real_mask = labels == 0
            fake_mask = labels == 1

            if real_mask.any():
                self._update_memory(name, contribution[real_mask].mean(dim=0), "real")
            if fake_mask.any():
                self._update_memory(name, contribution[fake_mask].mean(dim=0), "fake")

            return grad_output

        return hook

    # -------------------------
    # memory update
    # -------------------------
    def _update_memory(self, name, score, target):
        # 0=real/bonafide updates real memories; 1=fake/spoof updates fake memories.
        # Missing classes in a batch are not decayed.
        task_mem = self.task_mem_real if target == "real" else self.task_mem_fake
        self._ema_memory_update(task_mem, name, score)

    def _consolidate_task_memory(self):
        """Merge one task-level EMA into task-balanced historical memory."""
        previous_tasks = self.num_consolidated_tasks
        denominator = previous_tasks + 1

        for task_memory, global_memory in (
            (self.task_mem_real, self.global_mem_real),
            (self.task_mem_fake, self.global_mem_fake),
        ):
            for name, task_score in task_memory.items():
                task_score = task_score.detach()
                if name not in global_memory or previous_tasks == 0:
                    global_memory[name] = task_score.clone()
                    continue

                historical = global_memory[name].to(
                    device=task_score.device,
                    dtype=task_score.dtype,
                )
                global_memory[name] = (
                    previous_tasks * historical + task_score
                ) / denominator

        if self.task_mem_real or self.task_mem_fake:
            self.num_consolidated_tasks = denominator

    def _ema_memory_update(self, memory, name, score):
        score = score.detach()
        if name not in memory:
            memory[name] = score.clone()
            return
        memory[name] = self.mem_momentum * memory[name].to(score.device) + (1 - self.mem_momentum) * score

    def _norm(self, x):
        return x / (x.mean().detach() + self.eps)

    def get_role_probs(self, name, scope="task"):
        if scope == "global":
            mem_real = self.global_mem_real
            mem_fake = self.global_mem_fake
        else:
            mem_real = self.task_mem_real
            mem_fake = self.task_mem_fake

        if name not in mem_real or name not in mem_fake:
            return None

        R = self._norm(mem_real[name])
        F = self._norm(mem_fake[name])

        r = R / (R + self.role_activation_tau)
        f = F / (F + self.role_activation_tau)

        s_shared = r * f
        s_bona = r * (1.0 - f)
        s_spoof = (1.0 - r) * f
        s_free = (1.0 - r) * (1.0 - f)

        scores = torch.stack(
            [s_shared, s_bona, s_spoof, s_free],
            dim=0,
        ).clamp_min(self.eps)

        probs = scores.pow(1.0 / self.role_temperature)
        probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(self.eps)

        return {
            "shared": probs[0],
            "bona": probs[1],
            "spoof": probs[2],
            "free": probs[3],
            "R": R,
            "F": F,
            "r": r,
            "f": f,
        }

    def prepare_role_snapshot(self):
        self.current_role_snapshot = {}

        with torch.no_grad():
            for name in self.module_map:
                current_role = self.get_role_probs(name)
                historical_role = self.historical_role_snapshot.get(name)
                if current_role is None or historical_role is None:
                    continue

                old_importance = self._role_importance(historical_role)
                current_importance = self._role_importance(current_role)

                shared = old_importance * current_importance
                old_specific = old_importance * (1.0 - current_importance)
                new_specific = (1.0 - old_importance) * current_importance
                free = (1.0 - old_importance) * (1.0 - current_importance)

                self.current_role_snapshot[name] = {
                    "shared": shared.detach().clone(),
                    "old_specific": old_specific.detach().clone(),
                    "new_specific": new_specific.detach().clone(),
                    "free": free.detach().clone(),
                    # Legacy aliases for checkpoint/diagnostic consumers.
                    "bona": old_specific.detach().clone(),
                    "spoof": new_specific.detach().clone(),
                    "old_importance": old_importance.detach().clone(),
                    "current_importance": current_importance.detach().clone(),
                }

    def _role_importance(self, role):
        """Convert class saliency to discriminative layer-relative importance."""
        raw = torch.maximum(role["R"], role["F"]).clamp_min(self.eps)
        log_raw = raw.log()
        center = log_raw.mean().detach()
        spread = log_raw.std(unbiased=False).detach()
        spread_floor = max(self.eps, 1e-3)
        z_score = (log_raw - center) / spread.clamp_min(self.eps)
        importance = torch.sigmoid(z_score / self.role_temperature)
        return torch.where(
            spread > spread_floor,
            importance,
            torch.full_like(importance, 0.5),
        )

    def _get_active_role(self, name):
        return self.current_role_snapshot.get(name)

    # -------------------------
    # core IRNR (residual)
    # -------------------------
    def capture_task_gradients(self, task_loss):
        weights = []
        names = []
        for name, module in self.module_map.items():
            if module.weight is not None and module.weight.requires_grad:
                weights.append(module.weight)
                names.append(name)

        if not weights:
            return

        self.collect_role_memory = False
        try:
            task_grads = torch.autograd.grad(
                task_loss,
                weights,
                retain_graph=True,
                allow_unused=True,
            )
        finally:
            self.collect_role_memory = True

        for name, task_grad in zip(names, task_grads):
            if task_grad is None:
                continue

            task_grad = task_grad.detach()
            row_norm = task_grad.norm(dim=1, keepdim=True)
            valid_rows = row_norm > self.eps
            if not valid_rows.any():
                continue

            task_grad_direction = task_grad / row_norm.clamp_min(self.eps)
            self._update_grad_running(name, task_grad_direction, valid_rows)

    def warp_gradients(self, reference_grads=None):
        with torch.no_grad():
            for name, module in self.module_map.items():
                role = self._get_active_role(name)

                if module.weight is not None and module.weight.grad is not None:
                    raw_g = module.weight.grad.detach().clone()

                    g_ref = None
                    if reference_grads is not None:
                        g_ref = reference_grads.get(id(module.weight))
                    elif name in self.grad_ref:
                        # Legacy fallback for callers/checkpoints predating V12.
                        g_ref = self.grad_ref[name]

                    if role is not None and g_ref is not None:
                        g_ref = g_ref.to(device=raw_g.device, dtype=raw_g.dtype)
                        if g_ref.shape == raw_g.shape:
                            final_g = self._irnr_weight_gradient(name, raw_g, g_ref, role)
                            module.weight.grad.copy_(final_g)

    def _irnr_weight_gradient(self, name, raw_g, g_ref, role):
        # g_ref is normally the current-batch KD gradient: the local direction
        # that preserves old-model behavior at the current parameter point.
        # Only its direction is used when decomposing the CE gradient.
        # raw_g, g_ref, ref_direction: [out_features, in_features].
        ref_norm = g_ref.norm(dim=1, keepdim=True)
        valid_ref = ref_norm > self.eps
        ref_direction = torch.where(
            valid_ref,
            g_ref / ref_norm.clamp_min(self.eps),
            torch.zeros_like(g_ref),
        )

        proj_scalar = (raw_g * ref_direction).sum(dim=1, keepdim=True)
        raw_parallel_positive = torch.clamp(proj_scalar, min=0.0) * ref_direction
        raw_parallel_conflict = torch.clamp(proj_scalar, max=0.0) * ref_direction
        # Orthogonal only means perpendicular to the historical direction; it is
        # not inherently safe or guaranteed to be purely new knowledge.
        raw_orthogonal = raw_g - raw_parallel_positive - raw_parallel_conflict

        view_shape = [-1, 1]
        p_shared = role["shared"].view(view_shape)
        p_old = role["old_specific"].view(view_shape)
        p_new = role["new_specific"].view(view_shape)
        p_free = role["free"].view(view_shape)

        # Preserve useful plasticity. Only directions that truly oppose the
        # KD-retention direction are gated; new/free neurons are fully released.
        conflict_keep = (
            p_shared * self.conflict_scale_shared
            + p_old * self.conflict_scale_old
            + p_new
            + p_free
        )
        irnr_g = (
            raw_parallel_positive
            + raw_orthogonal
            + conflict_keep * raw_parallel_conflict
        )

        final_g = raw_g + self.alpha_irnr * (irnr_g - raw_g)
        final_g = torch.where(valid_ref.expand_as(raw_g), final_g, raw_g)

        final_proj_scalar = (final_g * ref_direction).sum(dim=1, keepdim=True)
        final_parallel_positive = (
            torch.clamp(final_proj_scalar, min=0.0) * ref_direction
        )
        final_parallel_conflict = (
            torch.clamp(final_proj_scalar, max=0.0) * ref_direction
        )
        final_orthogonal = (
            final_g - final_parallel_positive - final_parallel_conflict
        )

        self._record_gradient_diagnostics(
            name=name,
            raw_g=raw_g,
            g_ref=g_ref,
            irnr_g=irnr_g,
            final_g=final_g,
            raw_parallel_positive=raw_parallel_positive,
            raw_parallel_conflict=raw_parallel_conflict,
            raw_orthogonal=raw_orthogonal,
            final_parallel_positive=final_parallel_positive,
            final_parallel_conflict=final_parallel_conflict,
            final_orthogonal=final_orthogonal,
            conflict_keep=conflict_keep,
            valid_ref=valid_ref,
            proj_scalar=proj_scalar,
            role=role,
        )

        return final_g

    def _update_grad_running(self, name, task_grad_direction, valid_rows):
        task_grad_direction = task_grad_direction.detach()
        valid_rows = valid_rows.detach()
        if name not in self.grad_running or self.grad_running[name].shape != task_grad_direction.shape:
            self.grad_running[name] = torch.zeros_like(task_grad_direction)
            self.grad_running[name] = torch.where(
                valid_rows.expand_as(task_grad_direction),
                task_grad_direction,
                self.grad_running[name],
            )
            return

        running = self.grad_running[name].to(device=task_grad_direction.device, dtype=task_grad_direction.dtype)
        updated = (
            self.grad_momentum * running
            + (1 - self.grad_momentum) * task_grad_direction
        )
        self.grad_running[name] = torch.where(
            valid_rows.expand_as(updated),
            updated,
            running,
        )

    def begin_task(self):
        self.historical_role_snapshot = {}
        with torch.no_grad():
            for name in self.module_map:
                role = self.get_role_probs(name, scope="global")
                if role is not None:
                    self.historical_role_snapshot[name] = {
                        key: value.detach().clone()
                        for key, value in role.items()
                    }
        self.grad_running = {}
        self.task_mem_real = {}
        self.task_mem_fake = {}
        self.diagnostics = {}
        self.current_role_snapshot = {}
        self.diagnostic_context = {}
        self.gradient_component_records = []
        self.task_projection_hits = {}
        self.task_projection_checks = {}
        self.task_projection_scale_sum = {}
        self.task_projection_scale_min = {}
        self.last_violations.clear()
        self.epoch_violations.clear()

    # -------------------------
    # Task anchor and legacy gradient-reference update
    # -------------------------
    def update_task_anchor(self):
        with torch.no_grad():
            self._consolidate_task_memory()

            for name, module in self.module_map.items():
                if module.weight is None:
                    continue

                self.anchor_weights[name] = module.weight.detach().clone()

                if name in self.grad_running:
                    self.grad_ref[name] = self._normalize_reference_direction(self.grad_running[name])

                role_info = self.get_role_probs(name)
                if role_info is not None:
                    self.prev_role_probs[name] = {
                        "shared": role_info["shared"].detach().clone(),
                        "bona": role_info["bona"].detach().clone(),
                        "spoof": role_info["spoof"].detach().clone(),
                        "free": role_info["free"].detach().clone(),
                    }

            for name, lam in list(self.lagrange_multipliers.items()):
                self.lagrange_multipliers[name] = (lam * self.lagrange_task_decay).detach()

        self.grad_running = {}
        self.task_mem_real = {}
        self.task_mem_fake = {}
        self.cached_outputs.clear()
        self.current_role_snapshot = {}
        self.last_violations.clear()

    def _normalize_reference_direction(self, grad_running):
        grad_running = grad_running.detach()
        ref_norm = grad_running.norm(dim=1, keepdim=True)
        valid_ref = ref_norm > self.eps
        return torch.where(
            valid_ref,
            grad_running / ref_norm.clamp_min(self.eps),
            torch.zeros_like(grad_running),
        )

    # -------------------------
    # Lagrangian
    # -------------------------
    def _compute_role_delta(self, name, role_info):
        if "old_importance" in role_info and "current_importance" in role_info:
            delta = torch.abs(
                role_info["current_importance"] - role_info["old_importance"]
            )
            delta_relative = delta / (delta.mean().detach() + self.eps)
            delta_abs_gate = delta / (delta + self.delta_abs_tau)
            return delta_relative * delta_abs_gate

        if name not in self.prev_role_probs:
            return torch.zeros_like(role_info["free"])

        prev = self.prev_role_probs[name]
        prev_free = prev.get("free", prev.get("unstable"))
        delta = (
            torch.abs(role_info["shared"] - prev["shared"].to(role_info["shared"].device))
            + torch.abs(role_info["bona"] - prev["bona"].to(role_info["bona"].device))
            + torch.abs(role_info["spoof"] - prev["spoof"].to(role_info["spoof"].device))
            + torch.abs(role_info["free"] - prev_free.to(role_info["free"].device))
        ) / 4.0

        # delta_bar combines layer-relative role change and absolute-change
        # significance. This prevents tiny all-layer noise from being amplified.
        delta_relative = delta / (delta.mean().detach() + self.eps)
        delta_abs_gate = delta / (delta + self.delta_abs_tau)
        return delta_relative * delta_abs_gate

    def compute_lagrangian_reg(self):
        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        active_layers = 0
        self.last_violations = {}

        for name, module in self.module_map.items():
            if name not in self.anchor_weights:
                continue

            role_info = self._get_active_role(name)
            if role_info is None:
                continue

            W = module.weight
            W_anchor = self.anchor_weights[name].to(device=W.device, dtype=W.dtype)

            # Relative weight drift: one scalar for each output row.
            # IRNR defines a neuron's gradient vector as the corresponding
            # Linear.weight row; bias is intentionally left to ordinary training.
            diff_sq = (W - W_anchor).pow(2).mean(dim=1)
            # Raw anchor row scale before numerical protection.
            anchor_scale_raw = W_anchor.pow(2).mean(dim=1)
            # Protected denominator for relative drift normalization.
            anchor_scale = torch.clamp(anchor_scale_raw, min=self.drift_scale_floor)
            drift = diff_sq / anchor_scale

            # Retention strength is determined only by frozen history.
            q = role_info["old_importance"].to(W.device).clamp(0.0, 1.0)


            budget = self.get_drift_budget(
                name,
                role_info,
                W.device
            )

            violation = torch.relu(q.detach() * drift - budget.detach())

            if name not in self.lagrange_multipliers or self.lagrange_multipliers[name].shape != violation.shape:
                self.lagrange_multipliers[name] = torch.zeros_like(violation.detach())

            lam = self.lagrange_multipliers[name].to(device=W.device, dtype=W.dtype)
            # layer_loss = lam.detach() * violation + 0.5 * self.rho * violation.pow(2)

            layer_loss = (
                    lam.detach() * violation
                    + 0.5 * self.penalty_rho * violation.pow(2)
            )

            total = total + layer_loss.mean()
            # self.last_violations[name] = violation.detach()
            if name not in self.epoch_violations:
                self.epoch_violations[name] = []

            self.epoch_violations[name].append(
                violation.detach()
            )
            self._record_reg_diagnostics(
                drift=drift,
                q=q,
                budget=budget,
                q_times_drift=q.detach() * drift.detach(),
                violation=violation,
                lam=lam,
            )
            active_layers += 1

        return total / max(active_layers, 1)

    # -------------------------
    # ✅ COMPATIBILITY FIX (IMPORTANT)
    # -------------------------
    def before_forward(self, labels):
        self.current_labels = labels

    def after_step(self):
        self.current_labels = None
        self.cached_outputs.clear()
        self.current_role_snapshot = {}

    def update_lagrange_multipliers(self):
        with torch.no_grad():
            for name, violation in self.last_violations.items():
                if name not in self.lagrange_multipliers:
                    self.lagrange_multipliers[name] = torch.zeros_like(violation)

                lam = self.lagrange_multipliers[name].to(violation.device)

                # lam = torch.clamp(lam + self.rho * violation, min=0.0, max=self.lambda_max)

                lam = torch.clamp(
                    lam + self.lambda_lr * violation,
                    min=0.0,
                    max=self.lambda_max
                )

                self.lagrange_multipliers[name] = lam.detach().cpu()

            self.last_violations.clear()

    def record_loss_diagnostics(self, task_loss, reg_loss):
        if not self.diagnostics_enabled:
            return
        task_value = float(task_loss.detach().item())
        reg_value = float(reg_loss.detach().item())
        denom = abs(task_value) + self.eps
        self._diag_append("loss/reg_over_task", reg_value / denom)

    def _record_gradient_diagnostics(
        self,
        name,
        raw_g,
        g_ref,
        irnr_g,
        final_g,
        raw_parallel_positive,
        raw_parallel_conflict,
        raw_orthogonal,
        final_parallel_positive,
        final_parallel_conflict,
        final_orthogonal,
        conflict_keep,
        valid_ref,
        proj_scalar,
        role,
    ):
        if not self.diagnostics_enabled and not self.mechanism_data_enabled:
            return

        if self.diagnostics_enabled:
            self._diag_append("grad/raw_norm", raw_g.detach().norm().item())
            self._diag_append("grad/irnr_norm", irnr_g.detach().norm().item())
            self._diag_append("grad/final_norm", final_g.detach().norm().item())
            self._diag_append("raw/parallel_positive_norm", raw_parallel_positive.detach().norm().item())
            self._diag_append("raw/parallel_conflict_norm", raw_parallel_conflict.detach().norm().item())
            self._diag_append("raw/orthogonal_norm", raw_orthogonal.detach().norm().item())
            self._diag_append("post/parallel_positive_norm", final_parallel_positive.detach().norm().item())
            self._diag_append("post/parallel_conflict_norm", final_parallel_conflict.detach().norm().item())
            self._diag_append("post/orthogonal_norm", final_orthogonal.detach().norm().item())
            self._diag_append("grad/conflict_keep_mean", conflict_keep.detach().mean().item())

            valid_rows = valid_ref.squeeze(1)
            if valid_rows.any():
                self._diag_append(
                    "grad/conflict_row_ratio",
                    (proj_scalar.squeeze(1)[valid_rows] < 0).float().mean().item(),
                )
            self._diag_append(
                "grad/cosine_ref_pre",
                self._cosine_similarity_scalar(raw_g, g_ref),
            )
            self._diag_append(
                "grad/cosine_ref_post",
                self._cosine_similarity_scalar(final_g, g_ref),
            )

        self._record_gradient_component_row(
            name=name,
            raw_g=raw_g,
            g_ref=g_ref,
            final_g=final_g,
            raw_parallel_positive=raw_parallel_positive,
            raw_parallel_conflict=raw_parallel_conflict,
            raw_orthogonal=raw_orthogonal,
            final_parallel_positive=final_parallel_positive,
            final_parallel_conflict=final_parallel_conflict,
            final_orthogonal=final_orthogonal,
            conflict_keep=conflict_keep,
            valid_ref=valid_ref,
            proj_scalar=proj_scalar,
        )

        if not self.diagnostics_enabled:
            return

        role_names = ("shared", "old_specific", "new_specific", "free")
        role_matrix = torch.stack([role[name].detach() for name in role_names], dim=0)
        entropy = -(role_matrix * role_matrix.clamp_min(self.eps).log()).sum(dim=0)
        entropy = entropy / torch.log(torch.tensor(4.0, device=entropy.device))
        hard_roles = role_matrix.argmax(dim=0)

        self._diag_append("role/entropy_mean", entropy.mean().item())
        self._diag_append("role/max_probability_mean", role_matrix.max(dim=0).values.mean().item())
        for role_index, role_name in enumerate(role_names):
            self._diag_append(f"role/{role_name}_mean", role[role_name].detach().mean().item())
            self._diag_append(
                f"role/{role_name}_hard_ratio",
                (hard_roles == role_index).float().mean().item(),
            )
        self._diag_append("role/old_importance_mean", role["old_importance"].detach().mean().item())
        self._diag_append("role/current_importance_mean", role["current_importance"].detach().mean().item())

    @staticmethod
    def _norm_value(tensor):
        return float(tensor.detach().norm().item())

    def _safe_ratio(self, numerator, denominator):
        numerator = self._norm_value(numerator)
        denominator = self._norm_value(denominator)
        if denominator <= self.eps:
            return float("nan")
        return numerator / denominator

    def _cosine_similarity_scalar(self, first, second):
        first = first.detach().reshape(-1)
        second = second.detach().reshape(-1)
        denominator = first.norm() * second.norm()
        if denominator.item() <= self.eps:
            return 0.0
        return float(torch.dot(first, second).item() / denominator.item())

    def set_diagnostic_context(self, task_idx, task_name, epoch, step):
        self.diagnostic_context = {
            "task_idx": int(task_idx),
            "task": str(task_name),
            "epoch": int(epoch),
            "step": int(step),
        }

    def _record_gradient_component_row(
        self,
        name,
        raw_g,
        g_ref,
        final_g,
        raw_parallel_positive,
        raw_parallel_conflict,
        raw_orthogonal,
        final_parallel_positive,
        final_parallel_conflict,
        final_orthogonal,
        conflict_keep,
        valid_ref,
        proj_scalar,
    ):
        if not self.mechanism_data_enabled or not self.diagnostic_context:
            return
        if self.diagnostic_context["step"] % self.mechanism_data_interval != 0:
            return

        valid_rows = valid_ref.squeeze(1)
        conflict_row_ratio = 0.0
        if valid_rows.any():
            conflict_row_ratio = float(
                (proj_scalar.squeeze(1)[valid_rows] < 0).float().mean().item()
            )

        row = dict(self.diagnostic_context)
        row.update({
            "layer": name,
            "pre_total_norm": self._norm_value(raw_g),
            "post_total_norm": self._norm_value(final_g),
            "pre_parallel_conflict_norm": self._norm_value(raw_parallel_conflict),
            "post_parallel_conflict_norm": self._norm_value(final_parallel_conflict),
            "pre_parallel_positive_norm": self._norm_value(raw_parallel_positive),
            "post_parallel_positive_norm": self._norm_value(final_parallel_positive),
            "pre_orthogonal_norm": self._norm_value(raw_orthogonal),
            "post_orthogonal_norm": self._norm_value(final_orthogonal),
            "conflict_retention": self._safe_ratio(
                final_parallel_conflict, raw_parallel_conflict
            ),
            "positive_retention": self._safe_ratio(
                final_parallel_positive, raw_parallel_positive
            ),
            "orthogonal_retention": self._safe_ratio(
                final_orthogonal, raw_orthogonal
            ),
            "conflict_keep_mean": float(conflict_keep.detach().mean().item()),
            "conflict_row_ratio": conflict_row_ratio,
            "cosine_current_history_pre": self._cosine_similarity_scalar(raw_g, g_ref),
            "cosine_current_history_post": self._cosine_similarity_scalar(final_g, g_ref),
        })
        self.gradient_component_records.append(row)

    def pop_gradient_component_records(self):
        records = self.gradient_component_records
        self.gradient_component_records = []
        return records

    def get_neuron_diagnostics_snapshot(self):
        """Return one row per Linear output neuron for mechanism analysis."""
        rows = []
        role_keys = ("shared", "old_specific", "new_specific", "free")
        role_labels = (
            "shared",
            "historical_specific",
            "current_specific",
            "free",
        )

        with torch.no_grad():
            for name, module in self.module_map.items():
                historical_role = self.historical_role_snapshot.get(name)
                current_role = self.get_role_probs(name, scope="task")
                if historical_role is None or current_role is None:
                    continue

                old_importance = self._role_importance(historical_role)
                current_importance = self._role_importance(current_role)
                role = {
                    "shared": old_importance * current_importance,
                    "old_specific": old_importance * (1.0 - current_importance),
                    "new_specific": (1.0 - old_importance) * current_importance,
                    "free": (1.0 - old_importance) * (1.0 - current_importance),
                    "old_importance": old_importance,
                    "current_importance": current_importance,
                }
                role_matrix = torch.stack([role[key] for key in role_keys], dim=0)
                hard_roles = role_matrix.argmax(dim=0)

                W = module.weight.detach()
                anchor = self.anchor_weights.get(name)
                if anchor is None or anchor.shape != W.shape:
                    continue
                anchor = anchor.to(device=W.device, dtype=W.dtype)
                diff_sq = (W - anchor).pow(2).mean(dim=1)
                anchor_scale = torch.clamp(
                    anchor.pow(2).mean(dim=1),
                    min=self.drift_scale_floor,
                )
                drift = diff_sq / anchor_scale
                budget = self.get_drift_budget(name, role, W.device)
                weighted_drift = old_importance * drift
                violation_amount = torch.relu(weighted_drift - budget)

                projection_hits = self.task_projection_hits.get(name)
                if projection_hits is None:
                    projection_hits = torch.zeros_like(hard_roles, device="cpu")
                projection_hits = projection_hits.detach().cpu()
                projection_checks = int(self.task_projection_checks.get(name, 0))
                projection_scale_sum = self.task_projection_scale_sum.get(name)
                projection_scale_min = self.task_projection_scale_min.get(name)
                projection_scale_available = (
                    projection_scale_sum is not None
                    and projection_scale_min is not None
                )
                if not projection_scale_available:
                    projection_scale_sum = torch.zeros_like(
                        projection_hits,
                        dtype=torch.float32,
                    )
                    projection_scale_min = torch.ones_like(
                        projection_hits,
                        dtype=torch.float32,
                    )
                projection_scale_sum = projection_scale_sum.detach().cpu()
                projection_scale_min = projection_scale_min.detach().cpu()

                cpu_values = {
                    key: value.detach().float().cpu()
                    for key, value in role.items()
                }
                old_cpu = old_importance.detach().float().cpu()
                current_cpu = current_importance.detach().float().cpu()
                drift_cpu = drift.detach().float().cpu()
                weighted_cpu = weighted_drift.detach().float().cpu()
                budget_cpu = budget.detach().float().cpu()
                violation_cpu = violation_amount.detach().float().cpu()
                hard_cpu = hard_roles.detach().cpu()

                for neuron_id in range(W.shape[0]):
                    hit_count = int(projection_hits[neuron_id].item())
                    scale_mean = float("nan")
                    scale_minimum = float("nan")
                    if hit_count > 0 and projection_scale_available:
                        scale_mean = float(
                            projection_scale_sum[neuron_id].item() / hit_count
                        )
                        scale_minimum = float(
                            projection_scale_min[neuron_id].item()
                        )
                    rows.append({
                        "layer": name,
                        "neuron_id": neuron_id,
                        "old_importance": float(old_cpu[neuron_id].item()),
                        "current_importance": float(current_cpu[neuron_id].item()),
                        "p_shared": float(cpu_values["shared"][neuron_id].item()),
                        "p_historical_specific": float(cpu_values["old_specific"][neuron_id].item()),
                        "p_current_specific": float(cpu_values["new_specific"][neuron_id].item()),
                        "p_free": float(cpu_values["free"][neuron_id].item()),
                        "hard_role": role_labels[int(hard_cpu[neuron_id].item())],
                        "relative_drift": float(drift_cpu[neuron_id].item()),
                        "weighted_drift": float(weighted_cpu[neuron_id].item()),
                        "drift_budget": float(budget_cpu[neuron_id].item()),
                        "projection_budget": float(
                            self.projection_margin * budget_cpu[neuron_id].item()
                        ),
                        "violation_amount": float(violation_cpu[neuron_id].item()),
                        "violation": int(violation_cpu[neuron_id].item() > 0.0),
                        "hard_projected": int(hit_count > 0),
                        "hard_projection_count": hit_count,
                        "hard_projection_checks": projection_checks,
                        "hard_projection_ratio": (
                            hit_count / projection_checks if projection_checks else 0.0
                        ),
                        "projection_scale_mean": scale_mean,
                        "projection_scale_min": scale_minimum,
                    })

        return rows

    def _record_reg_diagnostics(self, drift, q, budget, q_times_drift, violation, lam):
        if not self.diagnostics_enabled:
            return

        self._diag_tensor("drift", drift.detach(), include_quantiles=True)
        self._diag_tensor("q", q.detach(), include_quantiles=False)
        self._diag_tensor("budget", budget.detach(), include_quantiles=False)
        self._diag_tensor("q_times_drift", q_times_drift.detach(), include_quantiles=True)
        self._diag_append("violation/mean", violation.detach().mean().item())
        self._diag_append("violation/max", violation.detach().max().item())
        self._diag_append("violation/nonzero_ratio", (violation.detach() > 0).float().mean().item())
        self._diag_append("lambda/mean", lam.detach().mean().item())
        self._diag_append("lambda/max", lam.detach().max().item())

    def _diag_tensor(self, prefix, tensor, include_quantiles):
        flat = tensor.flatten()
        if flat.numel() == 0:
            return
        self._diag_append(f"{prefix}/min", flat.min().item())
        self._diag_append(f"{prefix}/mean", flat.mean().item())
        self._diag_append(f"{prefix}/max", flat.max().item())
        if include_quantiles:
            quantiles = torch.quantile(
                flat.float(),
                torch.tensor([0.5, 0.9, 0.99], device=flat.device),
            )
            self._diag_append(f"{prefix}/p50", quantiles[0].item())
            self._diag_append(f"{prefix}/p90", quantiles[1].item())
            self._diag_append(f"{prefix}/p99", quantiles[2].item())

    def _diag_append(self, key, value):
        self.diagnostics.setdefault(key, []).append(float(value))

    def get_diagnostics_summary(self, reset=False):
        summary = {}
        for key, values in self.diagnostics.items():
            if values:
                summary[key] = float(sum(values) / len(values))
        if reset:
            self.diagnostics = {}
        return summary

    def get_state(self):
        return {
            "global_mem_real": copy.deepcopy(self.global_mem_real),
            "global_mem_fake": copy.deepcopy(self.global_mem_fake),
            "num_consolidated_tasks": self.num_consolidated_tasks,
            "task_mem_real": copy.deepcopy(self.task_mem_real),
            "task_mem_fake": copy.deepcopy(self.task_mem_fake),
            "grad_ref": copy.deepcopy(self.grad_ref),
            "grad_running": copy.deepcopy(self.grad_running),
            "prev_role_probs": copy.deepcopy(self.prev_role_probs),
            "lagrange_multipliers": copy.deepcopy(self.lagrange_multipliers),
            "anchor_weights": copy.deepcopy(self.anchor_weights),
        }

    def load_state(self, state):
        self.global_mem_real = copy.deepcopy(state.get("global_mem_real", state.get("mem_real", {})))
        self.global_mem_fake = copy.deepcopy(state.get("global_mem_fake", state.get("mem_fake", {})))
        default_task_count = 1 if self.global_mem_real or self.global_mem_fake else 0
        self.num_consolidated_tasks = int(state.get("num_consolidated_tasks", default_task_count))
        self.task_mem_real = copy.deepcopy(state.get("task_mem_real", {}))
        self.task_mem_fake = copy.deepcopy(state.get("task_mem_fake", {}))
        self.grad_ref = copy.deepcopy(state.get("grad_ref", {}))
        self.grad_running = copy.deepcopy(state.get("grad_running", {}))
        self.prev_role_probs = copy.deepcopy(state.get("prev_role_probs", {}))
        for layer_state in self.prev_role_probs.values():
            if "free" not in layer_state and "unstable" in layer_state:
                layer_state["free"] = layer_state.pop("unstable")
        self.lagrange_multipliers = copy.deepcopy(state.get("lagrange_multipliers", {}))
        self.anchor_weights = copy.deepcopy(state.get("anchor_weights", {}))
        self.current_role_snapshot = {}
        self.historical_role_snapshot = {}

    def reset_epoch_violations(self):
        self.epoch_violations = {}


    def update_lagrange_multipliers_epoch(self):

        with torch.no_grad():

            for name, violations in self.epoch_violations.items():

                mean_violation = torch.stack(
                    violations,
                    dim=0
                ).mean(dim=0)

                lam = self.lagrange_multipliers[name].to(
                    mean_violation.device
                )

                old_lam = lam.clone()

                lam = torch.clamp(
                    lam + self.lambda_lr * mean_violation,
                    min=0.0,
                    max=self.lambda_max
                )

                if self.diagnostics_enabled:
                    self._diag_append(
                        f"lambda_epoch/{name}/max",
                        lam.max().item()
                    )

                    self._diag_append(
                        f"lambda_epoch/{name}/mean",
                        lam.mean().item()
                    )

                    self._diag_append(
                        f"lambda_update/{name}/max",
                        (lam - old_lam).max().item()
                    )

                self.lagrange_multipliers[name] = (
                    lam.detach().cpu()
                )

        self.epoch_violations = {}

    def get_drift_budget(self, name, role_info, device):
        delta_bar = self._compute_role_delta(
            name,
            role_info
        ).to(device)

        delta_capped = torch.clamp(
            delta_bar,
            0.0,
            self.delta_max
        )

        delta_relax = delta_capped / (
                1.0 + delta_capped
        )

        budget = self.eps0 * (
                1.0 + self.lambda_delta * delta_relax
        )

        return budget

    def project_drift_constraint(self):

        with torch.no_grad():

            for name, module in self.module_map.items():

                if name not in self.anchor_weights:
                    continue

                if module.weight is None:
                    continue

                W = module.weight

                anchor = self.anchor_weights[name].to(
                    device=W.device,
                    dtype=W.dtype
                )

                role_info = self._get_active_role(name)

                if role_info is None:
                    continue

                # =====================================
                # 1. relative row-wise drift
                # =====================================
                diff_sq = (W - anchor).pow(2).mean(dim=1)

                anchor_scale = torch.clamp(
                    anchor.pow(2).mean(dim=1),
                    min=self.drift_scale_floor
                )

                drift = diff_sq / anchor_scale

                # =====================================
                # 2. 与 Lagrangian 完全一致的 q
                # =====================================
                q = role_info["old_importance"].to(W.device).clamp(0.0, 1.0)

                # =====================================
                # 3. 与 Lagrangian 相同的 soft budget
                # =====================================
                budget = self.get_drift_budget(
                    name,
                    role_info,
                    W.device
                )

                weighted_drift = q.detach() * drift

                # =====================================
                # 4. projection 只作为更外层的硬兜底
                #
                # Lagrangian:
                #       q * drift <= budget
                #
                # Projection:
                #       q * drift <= 1.2 * budget
                # =====================================
                projection_budget = self.projection_margin * budget

                mask = weighted_drift > projection_budget

                if self.mechanism_data_enabled:
                    hits = self.task_projection_hits.get(name)
                    if hits is None or hits.numel() != mask.numel():
                        hits = torch.zeros(mask.numel(), dtype=torch.long)
                    self.task_projection_hits[name] = (
                        hits + mask.detach().to(device="cpu", dtype=torch.long)
                    )
                    self.task_projection_checks[name] = (
                        self.task_projection_checks.get(name, 0) + 1
                    )

                if self.diagnostics_enabled:
                    self._diag_append(
                        "projection/nonzero_ratio",
                        mask.float().mean().item()
                    )

                if mask.any():

                    # 因为 drift 与 ||W-anchor||^2 成正比，
                    # 所以参数距离需要按 sqrt 比例缩放。
                    scale = torch.sqrt(
                        projection_budget[mask]
                        /
                        (weighted_drift[mask] + self.eps)
                    )

                    if self.mechanism_data_enabled:
                        scale_cpu = scale.detach().float().cpu()
                        mask_cpu = mask.detach().cpu()
                        scale_sum = self.task_projection_scale_sum.get(name)
                        scale_min = self.task_projection_scale_min.get(name)
                        if scale_sum is None or scale_sum.numel() != mask.numel():
                            scale_sum = torch.zeros(mask.numel(), dtype=torch.float32)
                            scale_min = torch.ones(mask.numel(), dtype=torch.float32)
                        scale_sum[mask_cpu] += scale_cpu
                        scale_min[mask_cpu] = torch.minimum(
                            scale_min[mask_cpu],
                            scale_cpu,
                        )
                        self.task_projection_scale_sum[name] = scale_sum
                        self.task_projection_scale_min[name] = scale_min

                    # ===== projection diagnostics =====
                    if self.diagnostics_enabled:
                        self._diag_append(
                            "projection/scale_mean",
                            scale.mean().item()
                        )

                        self._diag_append(
                            "projection/scale_min",
                            scale.min().item()
                        )

                        self._diag_append(
                            "projection/weighted_drift_mean",
                            weighted_drift.mean().item()
                        )

                        self._diag_append(
                            "projection/budget_mean",
                            projection_budget.mean().item()
                        )

                    new_W = W.clone()

                    new_W[mask] = (
                            anchor[mask]
                            +
                            (W[mask] - anchor[mask])
                            *
                            scale.unsqueeze(1)
                    )

                    W.copy_(new_W)

                else:

                    if self.diagnostics_enabled:
                        self._diag_append(
                            "projection/weighted_drift_mean",
                            weighted_drift.mean().item()
                        )

                        self._diag_append(
                            "projection/budget_mean",
                            projection_budget.mean().item()
                        )
