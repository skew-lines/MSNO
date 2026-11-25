import copy
import random
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedReservoir:
    """
    Class-balanced reservoir memory.

    The total capacity is split across classes. For binary bonafide/spoof tasks,
    this avoids replay memory being dominated by the majority fake/spoof class.
    Samples are stored on CPU.
    """

    def __init__(self, max_size: int, seed: int = 1234, num_classes: int = 2):
        self.max_size = max_size
        self.num_classes = num_classes
        self.rng = random.Random(seed)
        self.samples: Dict[int, List[Tuple[torch.Tensor, int]]] = {
            c: [] for c in range(num_classes)
        }
        self.seen: Dict[int, int] = {c: 0 for c in range(num_classes)}

        if max_size <= 0:
            self.capacity_per_class = {c: 0 for c in range(num_classes)}
        else:
            base = max_size // num_classes
            rem = max_size % num_classes
            self.capacity_per_class = {
                c: base + (1 if c < rem else 0) for c in range(num_classes)
            }

    def __len__(self):
        return sum(len(v) for v in self.samples.values())

    def add_sample(self, item_x: torch.Tensor, item_y: int):
        if self.max_size <= 0:
            return

        label = int(item_y)
        if label not in self.samples:
            raise ValueError(
                f"Unexpected label={label}. "
                f"Expected labels in [0, {self.num_classes - 1}]."
            )

        capacity = self.capacity_per_class.get(label, 0)
        if capacity <= 0:
            return

        memory = self.samples[label]
        self.seen[label] += 1
        sample = (item_x.detach().cpu().clone(), label)

        if len(memory) < capacity:
            memory.append(sample)
        else:
            replace_idx = self.rng.randrange(self.seen[label])
            if replace_idx < capacity:
                memory[replace_idx] = sample

    def add_from_loader(self, loader, device: torch.device, max_batches: Optional[int] = None):
        if self.max_size <= 0:
            return

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                for item_x, item_y in zip(x, y):
                    self.add_sample(item_x, int(item_y.item()))

    def sample(self, batch_size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if len(self) == 0 or batch_size <= 0:
            return None

        non_empty_labels = [label for label, memory in self.samples.items() if memory]
        if not non_empty_labels:
            return None

        # Balanced sampling across available classes. If a class has fewer samples,
        # sampling is with replacement, consistent with replay mini-batch sampling.
        per_class = max(1, batch_size // len(non_empty_labels))
        remainder = batch_size % len(non_empty_labels)

        batch: List[Tuple[torch.Tensor, int]] = []
        for idx, label in enumerate(non_empty_labels):
            take = per_class + (1 if idx < remainder else 0)
            memory = self.samples[label]
            for _ in range(take):
                batch.append(self.rng.choice(memory))

        self.rng.shuffle(batch)
        batch = batch[:batch_size]

        if not batch:
            return None

        x = torch.stack([item[0] for item in batch]).to(device)
        y = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)

        return x, y


# Backward-compatible alias used by ER and A-GEM.
ReservoirBuffer = ClassBalancedReservoir


class PerTaskEpisodicMemory:
    """
    Per-task class-balanced episodic memory.

    Closer to original GEM: each previous task has its own memory and contributes
    a separate gradient constraint. Within each task, memory is balanced by class.
    """

    def __init__(self, memory_per_task: int, seed: int = 1234, num_classes: int = 2):
        self.memory_per_task = memory_per_task
        self.seed = seed
        self.num_classes = num_classes
        self.memories: Dict[int, ClassBalancedReservoir] = {}

    def task_ids(self) -> List[int]:
        return sorted(self.memories.keys())

    def add_from_loader(self, task_id: int, loader, device: torch.device, max_batches: Optional[int] = None):
        if self.memory_per_task <= 0:
            return

        if task_id not in self.memories:
            self.memories[task_id] = ClassBalancedReservoir(
                self.memory_per_task,
                seed=self.seed + int(task_id),
                num_classes=self.num_classes,
            )

        self.memories[task_id].add_from_loader(loader, device, max_batches=max_batches)

    def sample_task(self, task_id: int, batch_size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        memory = self.memories.get(task_id)
        if memory is None:
            return None
        return memory.sample(batch_size, device)


class ContinualStrategy:
    needs_replay = False

    def __init__(self, model: nn.Module, args, device: torch.device):
        self.model = model
        self.args = args
        self.device = device

        # 当前任务 loss 使用这个 criterion。
        # 主训练脚本会通过 set_class_weights() 给它设置当前任务的 class weight。
        self.criterion = nn.CrossEntropyLoss()

        # replay / memory loss 使用这个 criterion。
        # 它不带 class weight，避免把“当前任务”的类别权重错误套到旧任务 memory 上。
        self.replay_criterion = nn.CrossEntropyLoss()

    def set_class_weights(self, weights: Optional[torch.Tensor]):
        if weights is not None:
            weights = weights.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # 注意：不要在这里给 replay_criterion 设置当前任务 class weight。
        # replay / memory loss 始终保持无权重 CE，避免当前任务类别分布影响旧任务约束。
        self.replay_criterion = nn.CrossEntropyLoss()

    def before_task(self, task_id: int):
        pass

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, y)

    def backward(self, loss: torch.Tensor):
        """Backpropagate the strategy objective."""
        loss.backward()

    def after_backward(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_task(self, task_id: int, loader=None):
        pass

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Optional[Dict]):
        del state


class FineTuningStrategy(ContinualStrategy):
    """
    Naive sequential fine-tuning.
    A single model is trained through tasks sequentially with no anti-forgetting mechanism.
    """
    pass


class ERStrategy(ContinualStrategy):
    """
    Experience Replay.
    Main loss = CE(current batch) + replay_weight * CE(memory batch).

    当前任务 batch:
        使用 self.criterion，可带当前任务 class weight。

    replay memory batch:
        使用 self.replay_criterion，不带当前任务 class weight。
    """

    needs_replay = True

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.buffer = ReservoirBuffer(args.memory_size, args.seed)

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        current_loss = self.criterion(logits, y)

        replay_batch = self.buffer.sample(
            min(x.size(0), self.args.replay_batch_size),
            self.device,
        )

        if replay_batch is None:
            return current_loss

        replay_x, replay_y = replay_batch
        replay_logits = self.model(replay_x)

        # 关键修改：memory loss 不使用当前任务 class weight
        replay_loss = self.replay_criterion(replay_logits, replay_y)

        return current_loss + self.args.replay_weight * replay_loss

    def after_task(self, task_id: int, loader=None):
        if loader is not None:
            self.buffer.add_from_loader(loader, self.device)


class LwFStrategy(ContinualStrategy):
    """
    Learning without Forgetting.
    For this fixed binary output-space setting, distillation is applied to all logits.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.teacher: Optional[nn.Module] = None

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        ce_loss = self.criterion(logits, y)

        if self.teacher is None:
            return ce_loss

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        temperature = self.args.lwf_temperature

        distill_loss = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

        return ce_loss + self.args.lwf_alpha * distill_loss

    def after_task(self, task_id: int, loader=None):
        self.teacher = copy.deepcopy(self.model).to(self.device)
        self.teacher.eval()

        for param in self.teacher.parameters():
            param.requires_grad = False


class EWCStrategy(ContinualStrategy):
    """
    Original-style EWC baseline.
    Keeps one parameter snapshot and one diagonal empirical Fisher per previous task.
    The regularization penalty sums over all previous tasks.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.prev_params_list: List[Dict[str, torch.Tensor]] = []
        self.fisher_list: List[Dict[str, torch.Tensor]] = []

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        ce_loss = self.criterion(logits, y)

        if not self.prev_params_list:
            return ce_loss

        penalty = torch.zeros((), device=self.device)

        for prev_params, fisher in zip(self.prev_params_list, self.fisher_list):
            for name, param in trainable_named_parameters(self.model):
                if name in prev_params and name in fisher:
                    penalty = penalty + (
                        fisher[name] * (param - prev_params[name]).pow(2)
                    ).sum()

        return ce_loss + 0.5 * self.args.ewc_lambda * penalty

    def after_task(self, task_id: int, loader=None):
        if loader is None:
            return

        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in trainable_named_parameters(self.model)
        }

        was_training = self.model.training
        self.model.eval()
        num_batches = 0

        for x, y in loader:
            if num_batches >= self.args.importance_batches:
                break

            x = x.to(self.device)
            y = y.long().to(self.device)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            # Fisher estimation uses unweighted CE for a cleaner standard EWC baseline.
            loss = self.replay_criterion(logits, y)
            loss.backward()

            for name, param in trainable_named_parameters(self.model):
                if param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)

            num_batches += 1

        denom = max(num_batches, 1)

        for name in fisher:
            fisher[name] /= denom

        prev_params = {
            name: param.detach().clone()
            for name, param in trainable_named_parameters(self.model)
        }

        self.fisher_list.append(fisher)
        self.prev_params_list.append(prev_params)

        if was_training:
            self.model.train()


class SIStrategy(ContinualStrategy):
    """
    Synaptic Intelligence.
    Accumulates a path integral during training and converts it into omega after each task.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.omega: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.start_params: Dict[str, torch.Tensor] = {}
        self.last_params: Dict[str, torch.Tensor] = {}
        self.path_integral: Dict[str, torch.Tensor] = {}

    def before_task(self, task_id: int):
        self.start_params = {
            name: param.detach().clone()
            for name, param in trainable_named_parameters(self.model)
        }

        self.last_params = {
            name: param.detach().clone()
            for name, param in trainable_named_parameters(self.model)
        }

        self.path_integral = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in trainable_named_parameters(self.model)
        }

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        ce_loss = self.criterion(logits, y)

        if not self.prev_params:
            return ce_loss

        penalty = torch.zeros((), device=self.device)

        for name, param in trainable_named_parameters(self.model):
            if name in self.prev_params and name in self.omega:
                penalty = penalty + (
                    self.omega[name] * (param - self.prev_params[name]).pow(2)
                ).sum()

        return ce_loss + self.args.si_lambda * penalty

    def before_step(self):
        for name, param in trainable_named_parameters(self.model):
            if param.grad is not None and name in self.last_params:
                delta = param.detach() - self.last_params[name]
                self.path_integral[name] += -param.grad.detach() * delta

    def after_step(self):
        self.last_params = {
            name: param.detach().clone()
            for name, param in trainable_named_parameters(self.model)
        }

    def after_task(self, task_id: int, loader=None):
        for name, param in trainable_named_parameters(self.model):
            if name not in self.start_params:
                continue

            delta = param.detach() - self.start_params[name]
            contribution = self.path_integral[name] / (delta.pow(2) + self.args.si_eps)
            contribution = contribution.clamp(min=0)

            if name in self.omega:
                self.omega[name] += contribution
            else:
                self.omega[name] = contribution

        self.prev_params = {
            name: param.detach().clone()
            for name, param in trainable_named_parameters(self.model)
        }


class MASStrategy(ContinualStrategy):
    """
    Memory Aware Synapses.
    Estimates parameter importance from output-function sensitivity using unlabeled inputs.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.omega: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        ce_loss = self.criterion(logits, y)

        if not self.prev_params:
            return ce_loss

        penalty = torch.zeros((), device=self.device)

        for name, param in trainable_named_parameters(self.model):
            if name in self.prev_params and name in self.omega:
                penalty = penalty + (
                    self.omega[name] * (param - self.prev_params[name]).pow(2)
                ).sum()

        return ce_loss + 0.5 * self.args.mas_lambda * penalty

    def after_task(self, task_id: int, loader=None):
        if loader is None:
            return

        importance = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in trainable_named_parameters(self.model)
        }

        was_training = self.model.training
        self.model.eval()
        num_batches = 0

        for x, _ in loader:
            if num_batches >= self.args.importance_batches:
                break

            x = x.to(self.device)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)

            # Output-function sensitivity objective used by MAS-style implementations.
            output_norm = logits.norm(2, dim=1).mean()
            output_norm.backward()

            for name, param in trainable_named_parameters(self.model):
                if param.grad is not None:
                    importance[name] += param.grad.detach().abs()

            num_batches += 1

        denom = max(num_batches, 1)

        for name in importance:
            importance[name] /= denom

            if name in self.omega:
                self.omega[name] += importance[name]
            else:
                self.omega[name] = importance[name]

        self.prev_params = {
            name: param.detach().clone()
            for name, param in trainable_named_parameters(self.model)
        }

        if was_training:
            self.model.train()


class AGEMStrategy(ContinualStrategy):
    """
    Averaged GEM.
    Uses one averaged episodic-memory gradient constraint.

    当前任务梯度:
        来自当前 batch，使用 self.criterion，可带当前任务 class weight。

    memory reference gradient:
        来自 replay memory，使用 self.replay_criterion，不带当前任务 class weight。
    """

    needs_replay = True

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.buffer = ReservoirBuffer(args.memory_size, args.seed)

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, y)

    def after_backward(self):
        replay_batch = self.buffer.sample(self.args.replay_batch_size, self.device)

        if replay_batch is None:
            return

        current_grad = get_flat_grad(self.model)

        replay_x, replay_y = replay_batch
        self.model.zero_grad(set_to_none=True)

        # 关键修改：memory gradient 不使用当前任务 class weight
        replay_loss = self.replay_criterion(self.model(replay_x), replay_y)
        replay_loss.backward()

        reference_grad = get_flat_grad(self.model)

        dot = torch.dot(current_grad, reference_grad)
        ref_norm_sq = reference_grad.dot(reference_grad)

        if dot < -self.args.gem_margin:
            projected = current_grad - (
                (dot + self.args.gem_margin) / (ref_norm_sq + 1e-12)
            ) * reference_grad
            set_flat_grad(self.model, projected)
        else:
            set_flat_grad(self.model, current_grad)

    def after_task(self, task_id: int, loader=None):
        if loader is not None:
            self.buffer.add_from_loader(loader, self.device)


class GEMStrategy(ContinualStrategy):
    """
    Original GEM-style implementation.
    Each previous task contributes one separate gradient constraint.
    Projection is solved as a small dual QP.

    Requires scipy. If scipy is unavailable or too slow, use --method agem.

    当前任务梯度:
        来自当前 batch，使用 self.criterion，可带当前任务 class weight。

    每个旧任务 memory gradient:
        来自旧任务 episodic memory，使用 self.replay_criterion，不带当前任务 class weight。
    """

    needs_replay = True

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)

        num_tasks = len([t for t in getattr(args, "train_tasks", "").split(",") if t.strip()])
        default_memory_per_task = max(1, args.memory_size // max(num_tasks, 1))

        memory_per_task = getattr(args, "memory_per_task", default_memory_per_task)
        if memory_per_task is None or memory_per_task <= 0:
            memory_per_task = default_memory_per_task

        self.memory = PerTaskEpisodicMemory(memory_per_task, args.seed)

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, y)

    def after_backward(self):
        previous_task_ids = self.memory.task_ids()

        if not previous_task_ids:
            return

        current_grad = get_flat_grad(self.model)

        memory_grads = []

        for old_task_id in previous_task_ids:
            replay_batch = self.memory.sample_task(old_task_id, self.args.replay_batch_size, self.device)

            if replay_batch is None:
                continue

            replay_x, replay_y = replay_batch

            self.model.zero_grad(set_to_none=True)

            # 关键修改：旧任务 memory gradient 不使用当前任务 class weight
            replay_loss = self.replay_criterion(self.model(replay_x), replay_y)
            replay_loss.backward()

            memory_grads.append(get_flat_grad(self.model))

        if not memory_grads:
            set_flat_grad(self.model, current_grad)
            return

        memory_grad_matrix = torch.stack(memory_grads, dim=0)

        violations = torch.mv(memory_grad_matrix, current_grad) < -self.args.gem_margin
        if not torch.any(violations):
            set_flat_grad(self.model, current_grad)
            return

        projected = project_gradient_qp(
            current_grad=current_grad,
            memory_grad_matrix=memory_grad_matrix,
            margin=self.args.gem_margin,
            max_iter=getattr(self.args, "gem_qp_max_iter", 200),
        )

        set_flat_grad(self.model, projected)

    def after_task(self, task_id: int, loader=None):
        if loader is not None:
            self.memory.add_from_loader(task_id, loader, self.device)


def _classifier_linear_layers(model: nn.Module) -> Tuple[nn.Module, List[nn.Linear]]:
    root = model.module if isinstance(model, nn.DataParallel) else model
    classifier = getattr(root, "classifier", None)
    if classifier is None:
        raise ValueError("Classifier-only strategies require model.classifier.")

    layers = [
        module
        for module in classifier.modules()
        if isinstance(module, nn.Linear) and module.weight.requires_grad
    ]
    if not layers:
        raise ValueError("A trainable Linear classifier layer is required.")
    return classifier, layers


class WeightModificationStrategy(ContinualStrategy):
    """Stable classifier-only projector shared by RAWM and RWM.

    Only the last Linear weight is modified. P is learned from that layer's
    input, so for the current classifier P has shape [1024, 1024]. The frozen
    XLS-R encoder and all bias terms remain untouched.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.task_id = -1
        self.alpha = float(args.wm_alpha)
        self.eps = float(args.wm_eps)
        if self.alpha <= 0:
            raise ValueError("--wm_alpha must be positive.")
        if self.eps <= 0:
            raise ValueError("--wm_eps must be positive.")

        self.classifier, linear_layers = _classifier_linear_layers(model)
        self.first_linear = linear_layers[0]
        self.target_linear = linear_layers[-1]
        self.P: Optional[torch.Tensor] = None
        self._last_classifier_input: Optional[torch.Tensor] = None
        self._last_layer_input: Optional[torch.Tensor] = None
        self._last_labels: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        self._hook_handles = [
            self.first_linear.register_forward_pre_hook(self._capture_classifier_input),
            self.target_linear.register_forward_pre_hook(self._capture_target_input),
        ]

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _capture_classifier_input(self, module, inputs):
        del module
        self._last_classifier_input = inputs[0].detach()

    def _capture_target_input(self, module, inputs):
        del module
        self._last_layer_input = inputs[0].detach()

    def before_task(self, task_id: int):
        self.task_id = int(task_id)

    def _ensure_projector(self):
        if self._last_layer_input is None:
            raise RuntimeError("Classifier input hook did not capture the final Linear input.")
        dim = int(self._last_layer_input.shape[-1])
        if self.P is None:
            self.P = torch.eye(dim, device=self.device, dtype=torch.float32)
        elif self.P.shape != (dim, dim):
            raise RuntimeError(
                f"Projector shape {tuple(self.P.shape)} does not match layer input dim {dim}."
            )

    @torch.no_grad()
    def _update_projector(self):
        self._ensure_projector()
        features = self._last_layer_input.detach().to(
            device=self.P.device, dtype=self.P.dtype
        )
        r = features.reshape(-1, features.shape[-1]).mean(dim=0)
        k = self.P.matmul(r)
        denom = self.alpha + torch.dot(r, k)
        denom = denom.clamp_min(self.eps)
        self.P.sub_(torch.outer(k, k) / denom)
        self.P.copy_(0.5 * (self.P + self.P.t()))

    def _direction(self, coefficient: torch.Tensor) -> torch.Tensor:
        self._ensure_projector()
        weight = self.target_linear.weight
        p = self.P.to(device=weight.device, dtype=weight.dtype)
        p_norm = p / p.norm(p="fro").clamp_min(self.eps)

        # P is the RLS soft orthogonal projector. I-P is the stable
        # complementary (previous-input) direction; this avoids the singular
        # inverse used by the source repository.
        q = torch.eye(p.shape[0], device=p.device, dtype=p.dtype) - p
        q_norm_value = q.norm(p="fro")
        if q_norm_value <= self.eps:
            q_norm = torch.zeros_like(q)
        else:
            q_norm = q / q_norm_value

        coefficient = torch.as_tensor(
            coefficient, device=weight.device, dtype=weight.dtype
        )
        return p_norm + coefficient * q_norm

    @torch.no_grad()
    def _project_target_gradient(self, coefficient: torch.Tensor):
        grad = self.target_linear.weight.grad
        if grad is None:
            return
        direction = self._direction(coefficient)
        grad.copy_(grad.matmul(direction.t()))

    def state_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "P": None if self.P is None else self.P.detach().cpu(),
        }

    def load_state_dict(self, state: Optional[Dict]):
        if not state:
            return
        self.task_id = int(state.get("task_id", -1))
        projector = state.get("P")
        self.P = None if projector is None else projector.to(
            self.device, dtype=torch.float32
        )


class RAWMStrategy(WeightModificationStrategy):
    """RAWM adapted to the frozen-XLS-R classifier-only protocol.

    CE and historical classifier distillation gradients are separated. On
    tasks after the first, only the CE gradient of classifier[3].weight is
    projected; the KD gradient remains unprojected, matching RAWM's intended
    regularization role. The teacher contains only the small classifier because
    the shared XLS-R encoder is frozen.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        if not getattr(args, "freeze_ssl", True):
            raise ValueError(
                "This RAWM adaptation stores a classifier-only teacher and therefore "
                "requires the shared XLS-R encoder to be frozen."
            )
        super().__init__(model, args, device)
        self.m = float(args.rawm_m)
        self.eta = float(args.rawm_eta)
        self.temperature = float(args.rawm_temperature)
        if self.m < 0:
            raise ValueError("--rawm_m must be non-negative.")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("--rawm_eta must be in [0, 1].")
        if self.temperature <= 0:
            raise ValueError("--rawm_temperature must be positive.")

        self.teacher_classifier: Optional[nn.Module] = None
        self._last_ce: Optional[torch.Tensor] = None
        self._last_reg: Optional[torch.Tensor] = None
        self._used_custom_backward = False

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        del x
        self._last_labels = y.detach()
        self._last_ce = self.criterion(logits, y)
        self._last_reg = None

        if self.teacher_classifier is None:
            return self._last_ce
        if self._last_classifier_input is None:
            raise RuntimeError("RAWM did not capture the pooled classifier input.")

        with torch.no_grad():
            teacher_logits = self.teacher_classifier(self._last_classifier_input)
        temperature = self.temperature
        self._last_reg = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)
        return (1.0 - self.eta) * self._last_ce + self.eta * self._last_reg

    def _rawm_coefficient(self) -> torch.Tensor:
        counts = torch.bincount(self._last_labels.long(), minlength=2)
        beta = (counts[0].float() + 1.0) / (counts[1].float() + 1.0)
        return self.m * beta

    def backward(self, loss: torch.Tensor):
        if self.teacher_classifier is None or self._last_reg is None:
            self._used_custom_backward = False
            loss.backward()
            return

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        ce_grads = torch.autograd.grad(
            self._last_ce, parameters, retain_graph=True, allow_unused=True
        )
        reg_grads = torch.autograd.grad(
            self._last_reg, parameters, allow_unused=True
        )
        direction = self._direction(self._rawm_coefficient())

        for parameter, ce_grad, reg_grad in zip(parameters, ce_grads, reg_grads):
            if ce_grad is None and reg_grad is None:
                parameter.grad = None
                continue
            if ce_grad is None:
                ce_grad = torch.zeros_like(parameter)
            if reg_grad is None:
                reg_grad = torch.zeros_like(parameter)
            if parameter is self.target_linear.weight:
                ce_grad = ce_grad.matmul(direction.t())
            parameter.grad = (
                (1.0 - self.eta) * ce_grad + self.eta * reg_grad
            ).detach()
        self._used_custom_backward = True

    def after_backward(self):
        if not self._used_custom_backward and self.task_id > 0:
            self._project_target_gradient(self._rawm_coefficient())
        self._update_projector()
        self._used_custom_backward = False

    def _snapshot_classifier(self) -> nn.Module:
        self._remove_hooks()
        try:
            teacher = copy.deepcopy(self.classifier).to(self.device)
        finally:
            self._register_hooks()
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    def after_task(self, task_id: int, loader=None):
        del loader
        self.task_id = int(task_id)
        self.teacher_classifier = self._snapshot_classifier()

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["teacher_classifier"] = (
            None
            if self.teacher_classifier is None
            else self.teacher_classifier.state_dict()
        )
        return state

    def load_state_dict(self, state: Optional[Dict]):
        super().load_state_dict(state)
        if state and state.get("teacher_classifier") is not None:
            self.teacher_classifier = self._snapshot_classifier()
            self.teacher_classifier.load_state_dict(state["teacher_classifier"])


class RWMStrategy(WeightModificationStrategy):
    """Repository-compatible logit-radian RWM for the unified classifier.

    The reviewed repository has no trainable attention implementation. This
    adapter therefore uses its observable rule: correct-class logits are
    normalized across the batch and mapped to signed rotation radians
    (real=+1, fake=-1). It should be reported as the repository-compatible RWM
    variant, not as a reproduction of the paper's missing SA module.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        super().__init__(model, args, device)
        self.score_temperature = float(args.rwm_score_temperature)
        if self.score_temperature <= 0:
            raise ValueError("--rwm_score_temperature must be positive.")
        self._last_logits: Optional[torch.Tensor] = None

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        del x
        self._last_labels = y.detach()
        self._last_logits = logits.detach()
        return self.criterion(logits, y)

    def _rwm_beta(self) -> torch.Tensor:
        correct_logits = self._last_logits.gather(
            1, self._last_labels.long().unsqueeze(1)
        ).squeeze(1)
        delta = F.softmax(correct_logits / self.score_temperature, dim=0)
        theta = torch.asin(delta.clamp(min=0.0, max=1.0 - self.eps))
        signs = torch.where(
            self._last_labels.long() == 0,
            torch.ones_like(theta),
            -torch.ones_like(theta),
        )
        theta_f = torch.pi / 4.0 + 0.5 * torch.sum(theta * signs)
        theta_f = theta_f.clamp(
            min=self.eps, max=float(torch.pi / 2.0) - self.eps
        )
        return torch.tan(theta_f)

    def after_backward(self):
        if self.task_id > 0:
            self._project_target_gradient(self._rwm_beta())
        self._update_projector()


class RegOStrategy(ContinualStrategy):
    """Region-Based Optimization adapted to the frozen-XLS-R protocol.

    The official repository trains a single Linear classifier on extracted
    features and modifies only ``module.weight``. Here the equivalent target is
    the final ``Linear(1024, 2)`` classifier weight. Label 0 is real and label 1
    is fake, following UnifiedDataset and the current binary protocol.
    """

    def __init__(self, model: nn.Module, args, device: torch.device):
        if not getattr(args, "freeze_ssl", True):
            raise ValueError(
                "This RegO adaptation targets the classifier on frozen XLS-R "
                "features and therefore requires --freeze_ssl."
            )
        super().__init__(model, args, device)
        _, linear_layers = _classifier_linear_layers(model)
        self.target_linear = linear_layers[-1]
        self.task_id = -1
        self.ef_threshold = float(args.rego_ef_threshold)
        self.importance_quantile = float(args.rego_importance_quantile)
        self.eps = float(args.rego_eps)
        self.importance_batches = int(args.importance_batches)

        if self.ef_threshold < 0:
            raise ValueError("--rego_ef_threshold must be non-negative.")
        if not 0.0 <= self.importance_quantile <= 1.0:
            raise ValueError("--rego_importance_quantile must be in [0, 1].")
        if self.eps <= 0:
            raise ValueError("--rego_eps must be positive.")
        if self.importance_batches <= 0:
            raise ValueError("--importance_batches must be positive for RegO.")

        self._last_labels: Optional[torch.Tensor] = None
        self.old_gradient: Optional[torch.Tensor] = None
        self.real_filters: List[torch.Tensor] = []
        self.fake_filters: List[torch.Tensor] = []

    def before_task(self, task_id: int):
        self.task_id = int(task_id)

    def loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        del x
        self._last_labels = y.detach()
        return self.criterion(logits, y)

    @staticmethod
    def _ebbinghaus_weights(
        task_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Solve sum(exp(-i / S))=1 without the source SciPy dependency."""
        if task_count <= 0:
            return torch.empty(0, device=device, dtype=dtype)
        if task_count == 1:
            return torch.ones(1, device=device, dtype=dtype)

        indices = torch.arange(1, task_count + 1, dtype=torch.float64)
        low, high = 1e-8, 1.0
        while torch.exp(-indices / high).sum().item() < 1.0:
            high *= 2.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            if torch.exp(-indices / middle).sum().item() < 1.0:
                low = middle
            else:
                high = middle

        weights = torch.exp(-indices / high).flip(0)
        weights = weights / weights.sum().clamp_min(1e-12)
        return weights.to(device=device, dtype=dtype)

    @torch.no_grad()
    def after_backward(self):
        grad = self.target_linear.weight.grad
        if grad is None:
            return

        current_grad = grad.detach().clone()
        history_count = min(len(self.real_filters), len(self.fake_filters))
        if self.task_id > 0 and history_count > 0 and self.old_gradient is not None:
            weights = self._ebbinghaus_weights(
                history_count, current_grad.device, current_grad.dtype
            )
            retention_score = torch.zeros_like(current_grad)
            real_union = torch.zeros_like(current_grad, dtype=torch.bool)
            fake_union = torch.zeros_like(current_grad, dtype=torch.bool)

            for index in range(history_count):
                real_mask = self.real_filters[index].to(
                    device=current_grad.device, dtype=torch.bool
                )
                fake_mask = self.fake_filters[index].to(
                    device=current_grad.device, dtype=torch.bool
                )
                retention_score.add_(
                    torch.logical_or(real_mask, fake_mask).to(current_grad.dtype)
                    * weights[index]
                )
                real_union.logical_or_(real_mask)
                fake_union.logical_or_(fake_mask)

            retained = retention_score > self.ef_threshold
            region_code = (
                real_union.to(torch.int8)
                + 2 * fake_union.to(torch.int8)
            ) * retained.to(torch.int8)

            old_gradient = self.old_gradient.to(
                device=current_grad.device, dtype=current_grad.dtype
            )
            old_flat = old_gradient.reshape(-1)
            current_flat = current_grad.reshape(-1)
            parallel_scale = torch.dot(current_flat, old_flat) / (
                torch.dot(old_flat, old_flat) + self.eps
            )
            parallel_grad = parallel_scale * old_gradient
            orthogonal_grad = current_grad - parallel_grad

            labels = self._last_labels.long()
            batch_size = max(int(labels.numel()), 1)
            real_fraction = (labels == 0).sum().to(current_grad.dtype) / batch_size
            fake_fraction = (labels == 1).sum().to(current_grad.dtype) / batch_size
            shared_grad = (
                real_fraction * parallel_grad
                + fake_fraction * orthogonal_grad
            )

            modified_grad = torch.zeros_like(current_grad)
            modified_grad.add_(current_grad * (region_code == 0))
            modified_grad.add_(parallel_grad * (region_code == 1))
            modified_grad.add_(orthogonal_grad * (region_code == 2))
            modified_grad.add_(shared_grad * (region_code == 3))
            grad.copy_(modified_grad)

        self.old_gradient = grad.detach().clone()

    def after_task(self, task_id: int, loader=None):
        if loader is None:
            raise RuntimeError("RegO requires a consolidation loader after each task.")

        was_training = self.model.training
        self.model.eval()
        importance = {
            0: torch.zeros_like(self.target_linear.weight, device=self.device),
            1: torch.zeros_like(self.target_linear.weight, device=self.device),
        }
        sample_counts = {0: 0, 1: 0}

        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= self.importance_batches:
                break
            x = x.to(self.device)
            y = y.long().to(self.device)
            logits = self.model(x)
            present_labels = [label for label in (0, 1) if torch.any(y == label)]

            for label_index, label in enumerate(present_labels):
                mask = y == label
                class_loss = self.replay_criterion(logits[mask], y[mask])
                class_grad = torch.autograd.grad(
                    class_loss,
                    self.target_linear.weight,
                    retain_graph=label_index < len(present_labels) - 1,
                )[0]
                importance[label].add_(class_grad.detach().pow(2))
                sample_counts[label] += int(mask.sum().item())

        for label in (0, 1):
            importance[label].div_(max(sample_counts[label], 1))

        real_threshold = torch.quantile(
            importance[0].reshape(-1), self.importance_quantile
        )
        fake_threshold = torch.quantile(
            importance[1].reshape(-1), self.importance_quantile
        )
        self.real_filters.append((importance[0] > real_threshold).detach())
        self.fake_filters.append((importance[1] > fake_threshold).detach())
        self.task_id = int(task_id)
        self.model.zero_grad(set_to_none=True)

        if was_training:
            self.model.train()

    def state_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "old_gradient": (
                None if self.old_gradient is None else self.old_gradient.detach().cpu()
            ),
            "real_filters": [mask.detach().cpu() for mask in self.real_filters],
            "fake_filters": [mask.detach().cpu() for mask in self.fake_filters],
        }

    def load_state_dict(self, state: Optional[Dict]):
        if not state:
            return
        self.task_id = int(state.get("task_id", -1))
        old_gradient = state.get("old_gradient")
        self.old_gradient = (
            None
            if old_gradient is None
            else old_gradient.to(self.device, dtype=self.target_linear.weight.dtype)
        )
        self.real_filters = [
            mask.to(self.device, dtype=torch.bool)
            for mask in state.get("real_filters", [])
        ]
        self.fake_filters = [
            mask.to(self.device, dtype=torch.bool)
            for mask in state.get("fake_filters", [])
        ]


def project_gradient_qp(
    current_grad: torch.Tensor,
    memory_grad_matrix: torch.Tensor,
    margin: float = 0.0,
    max_iter: int = 200,
) -> torch.Tensor:
    """
    Project current_grad onto:
        memory_grad_i dot projected_grad >= -margin

    Primal:
        min_z 0.5 ||z - g||^2
        s.t. A z >= b, where b = -margin

    Dual:
        min_lambda 0.5 lambda^T A A^T lambda + (A g - b)^T lambda
        s.t. lambda >= 0

    Solution:
        z = g + A^T lambda
    """
    try:
        import numpy as np
        from scipy.optimize import minimize
    except Exception as exc:
        raise RuntimeError(
            "Original GEM requires solving a small QP. Install scipy, "
            "or use --method agem for the averaged GEM approximation."
        ) from exc

    device = current_grad.device
    dtype = current_grad.dtype

    A = memory_grad_matrix.detach().double().cpu()
    g = current_grad.detach().double().cpu()
    b = torch.full((A.size(0),), -float(margin), dtype=torch.double)

    C = A @ A.t()
    q = A @ g - b

    C_np = C.numpy()
    q_np = q.numpy()
    n_constraints = A.size(0)

    def objective(lam):
        return 0.5 * float(lam @ C_np @ lam) + float(q_np @ lam)

    def gradient(lam):
        return C_np @ lam + q_np

    result = minimize(
        objective,
        x0=np.zeros(n_constraints, dtype=np.float64),
        jac=gradient,
        bounds=[(0.0, None)] * n_constraints,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"GEM QP solver failed: {result.message}")

    lam = torch.from_numpy(result.x).double()
    projected = g + A.t().matmul(lam)

    return projected.to(device=device, dtype=dtype)


def build_strategy(method: str, model: nn.Module, args, device: torch.device) -> ContinualStrategy:
    method = method.lower()

    strategies = {
        "fine_tuning": FineTuningStrategy,
        "finetuning": FineTuningStrategy,
        "naive": FineTuningStrategy,

        "er": ERStrategy,
        "replay": ERStrategy,

        "lwf": LwFStrategy,
        "ewc": EWCStrategy,
        "si": SIStrategy,
        "mas": MASStrategy,

        "agem": AGEMStrategy,
        "a-gem": AGEMStrategy,

        "gem": GEMStrategy,
        "rawm": RAWMStrategy,
        "rwm": RWMStrategy,
        "rego": RegOStrategy,
    }

    if method not in strategies:
        raise ValueError(
            f"Unsupported baseline method: {method}. "
            f"Supported methods: {', '.join(sorted(strategies.keys()))}"
        )

    return strategies[method](model, args, device)


def trainable_named_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def get_flat_grad(model: nn.Module) -> torch.Tensor:
    chunks = []

    for _, param in trainable_named_parameters(model):
        if param.grad is None:
            chunks.append(torch.zeros_like(param).flatten())
        else:
            chunks.append(param.grad.detach().clone().flatten())

    if not chunks:
        return torch.empty(0)

    return torch.cat(chunks)


def set_flat_grad(model: nn.Module, flat_grad: torch.Tensor):
    offset = 0

    for _, param in trainable_named_parameters(model):
        numel = param.numel()
        param.grad = flat_grad[offset: offset + numel].view_as(param).clone()
        offset += numel