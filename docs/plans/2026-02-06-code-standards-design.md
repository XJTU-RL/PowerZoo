# PowerZoo Code Standards Brandbook

> Design document for unified code standards across the PowerZoo project.
> Created: 2026-02-06 | Branch: `refactor/code-standards`

---

## 1. Design Decisions

| Dimension | Decision | Rationale |
|-----------|----------|-----------|
| Goal | Full modernization | Upgrade entire codebase to Python 3.10+ style |
| Indentation | 4 spaces (no tabs) | PEP 8 standard; 93% of codebase already uses 4 spaces |
| Type hints | `from __future__ import annotations` + native syntax | Python 3.10+ style; less boilerplate than `typing` module |
| Docstrings | Pure English, Google style | Academic project convention; open-source friendly |
| Logging | Standard `logging.getLogger(__name__)` | Zero dependency; SmartGrid custom logger to be deprecated |
| `super()` | Modern `super().__init__()` | Python 3+ standard; cleaner than `super(Class, self)` |
| Env signatures | `(args: dict[str, Any], rank: int \| None = None)` | Unified interface across all 4 environments |
| Imports | Absolute only, sorted by group | Project root-based; no relative imports |
| Comments | English only | Consistent with docstring language decision |

---

## 2. File Template

Every Python file in the project MUST follow this template:

```python
# -*- coding: utf-8 -*-
"""One-line English summary of the module.

Detailed description if needed. Keep it concise.
Key components:
    - ClassName: Brief role description.
"""
from __future__ import annotations

# Group 1: Standard library
import copy
import logging
import os

# Group 2: Third-party
import numpy as np
import torch
import torch.nn as nn

# Group 3: Project internal
from algorithms.actors.on_policy_base import OnPolicyBase
from utils.envs_tools import check
from utils.models_tools import get_grad_norm

logger = logging.getLogger(__name__)
```

### Import Rules

- `from __future__ import annotations` MUST be the first import, immediately after the module docstring.
- Three groups separated by blank lines: standard library, third-party, project internal.
- Each group sorted alphabetically.
- Multi-symbol imports use parentheses with one symbol per line:
  ```python
  from utils.models_tools import (
      get_grad_norm,
      huber_loss,
      mse_loss,
  )
  ```
- **No relative imports** (`from .xxx import`). The only exception is `__init__.py` for package re-exports.

---

## 3. Class & Method Conventions

### 3.1 Class Definition

```python
class HAPPO(OnPolicyBase):
    """HAPPO algorithm for heterogeneous-agent proximal policy optimization.

    Implements clipped surrogate objective with sequential agent updates
    and importance sampling weight propagation.

    Attributes:
        clip_param: Clipping parameter for PPO surrogate objective.
        ppo_epoch: Number of PPO training epochs per update.
    """

    def __init__(
        self,
        args: dict[str, Any],
        obs_space: gym.Space,
        act_space: gym.Space,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize HAPPO algorithm.

        Args:
            args: Algorithm configuration dictionary.
            obs_space: Observation space for the agent.
            act_space: Action space for the agent.
            device: Torch device for tensor operations.
        """
        super().__init__(args, obs_space, act_space, device)
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        logger.info(f"Initialized {self.__class__.__name__}")
```

### 3.2 Rules

- **Class names**: `PascalCase` (e.g., `OnPolicyBaseRunner`, `ContinuousQCritic`).
- **Algorithm acronyms**: Keep uppercase (e.g., `HAPPO`, `MADDPG`, `HASAC`).
- **Methods & functions**: `snake_case` (e.g., `get_actions`, `evaluate_actions`).
- **Private methods**: Leading underscore (e.g., `_collect_heterogeneous`, `_validate_config`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_GRAD_NORM = 0.5`).
- **Boolean attributes**: `use_` prefix (e.g., `self.use_recurrent_policy`).
- **Tensor config dict**: `self.tpdv = dict(dtype=torch.float32, device=device)` (project convention).
- **`super()` calls**: Always use modern `super().__init__()`, never `super(ClassName, self).__init__()`.
- **`__init__` return type**: Always annotate `-> None`.
- **Multi-parameter signatures**: Vertical layout with trailing comma.

### 3.3 Docstring Format

Google style, pure English:

```python
def update(self, sample: tuple) -> tuple[torch.Tensor, ...]:
    """Perform single gradient update on actor network.

    Args:
        sample: Batch data tuple containing (obs, rnn_states,
            actions, action_log_probs, advantages, masks).

    Returns:
        Tuple of (policy_loss, dist_entropy, actor_grad_norm, imp_weights).

    Raises:
        ValueError: If sample contains NaN values.
    """
```

- First line: imperative summary (e.g., "Perform...", "Compute...", "Initialize...").
- Parameter types go in the signature, NOT in the docstring (no `args: (dict) config` pattern).
- `Args:`, `Returns:`, `Raises:` sections as needed.
- Empty-body classes (e.g., `MADDPG(HADDPG): pass`) MUST explain why in the docstring.

---

## 4. Naming Conventions

### 4.1 Variable Naming

```python
# Instance attributes: snake_case
self.clip_param = args["clip_param"]
self.actor_optimizer = torch.optim.Adam(...)
self.use_max_grad_norm = args["use_max_grad_norm"]

# Local variables: snake_case
policy_loss = 0.0
obs_batch = sample[0]
is_training = True

# Constants: UPPER_SNAKE_CASE with _ separators for large numbers
MAX_GRAD_NORM = 0.5
DEFAULT_BUFFER_SIZE = 100_000
VALID_ACTION_SPACES = {"Box", "Discrete", "MultiDiscrete"}
```

### 4.2 Standard Abbreviations

These abbreviations are project-wide conventions and should be used consistently:

| Abbreviation | Full Name | Usage |
|-------------|-----------|-------|
| `obs` | observation | `obs_batch`, `share_obs`, `cent_obs` |
| `act` | action | `act_space`, `available_actions` |
| `rnn_states` | RNN hidden states | `rnn_states_actor`, `rnn_states_critic` |
| `tpdv` | tensor parameter dict | `self.tpdv = dict(dtype=..., device=...)` |
| `dist` | distribution | `dist_entropy`, `dist_loss` |
| `adv` | advantage | `adv_targ` |
| `imp` | importance sampling | `imp_weights` |
| `lr` | learning rate | `self.lr`, `lr_decay()` |
| `grad` | gradient | `actor_grad_norm`, `get_grad_norm()` |

---

## 5. Environment Module Standards

### 5.1 Unified Init Signature

All environment wrapper classes MUST use this signature:

```python
def __init__(
    self,
    args: dict[str, Any],
    rank: int | None = None,
) -> None:
```

- `args`: Configuration dictionary containing all environment parameters.
- `rank`: Worker index for parallel training (None for single-process).
- SmartGrid's current `(env, config, rank)` signature must be refactored to accept `args` and perform internal decomposition.

### 5.2 Required Methods

Every environment wrapper MUST implement:

```python
def step(self, actions: np.ndarray) -> tuple[list, list, list, list, list, list]:
    """Execute one environment step.

    Returns:
        Tuple of (obs, share_obs, rewards, dones, infos, avail_actions).
    """

def reset(self) -> tuple[list, list, list]:
    """Reset environment to initial state.

    Returns:
        Tuple of (obs, share_obs, avail_actions).
    """

def seed(self, seed: int) -> None:
    """Set random seed for reproducibility."""

def close(self) -> None:
    """Release environment resources."""

def _validate_config(self, args: dict[str, Any]) -> None:
    """Validate required configuration keys.

    Raises:
        KeyError: If required keys are missing.
        ValueError: If values are out of valid range.
    """
```

---

## 6. Error Handling & Configuration Access

### 6.1 Error Handling Patterns

```python
# Precondition checks: assert with clear message
assert (
    act_space.__class__.__name__ == "Box"
), f"Only continuous action space supported, got {act_space.__class__.__name__}"

# Data conversion: project-standard check() function
obs = check(obs).to(**self.tpdv)
rewards = check(rewards).to(**self.tpdv)

# External I/O: try/except with logging + re-raise
try:
    dss_engine = dss.DSS()
    dss_engine.Text.Command = f"compile [{dss_file}]"
except Exception as e:
    logger.error(f"Failed to compile DSS file {dss_file}: {e}")
    raise
```

### 6.2 Rules

- **Never silently swallow exceptions** (`except: pass` is forbidden).
- Use `assert` for internal invariants and preconditions.
- Use `try/except` only for external I/O (file operations, DSS engine, network).
- Always log errors at `logger.error()` level before re-raising.

### 6.3 Configuration Access

```python
# Required keys: direct access (raises KeyError if missing)
self.clip_param = args["clip_param"]
self.lr = args["lr"]

# Optional keys: .get() with sensible default
self.render_mode = args.get("render_mode", None)
self.seed_value = args.get("seed", 42)
```

---

## 7. Logging Standards

### 7.1 Setup

Every module MUST have a logger:

```python
import logging

logger = logging.getLogger(__name__)
```

Placed after all imports, before any class/function definitions.

### 7.2 Log Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| `logger.info()` | Initialization, major state changes | `logger.info(f"Initialized {self.__class__.__name__}")` |
| `logger.debug()` | Training steps, intermediate values | `logger.debug(f"Update step {step}, loss={loss:.4f}")` |
| `logger.warning()` | Recoverable issues, fallbacks | `logger.warning(f"NaN detected in rewards, clipping")` |
| `logger.error()` | Failures before re-raise | `logger.error(f"Failed to load DSS: {e}")` |

### 7.3 SmartGrid Migration

SmartGrid's custom `get_logger` / `log_training_step` / `setup_training_logger` will be:
- `get_logger` -> replaced by standard `logging.getLogger(__name__)`
- `log_training_step` -> moved to `utils/` as a standalone function
- `setup_training_logger` -> moved to `utils/` as a standalone function
- `envs/smartgrid/logging/` module deprecated after migration

---

## 8. Comment Standards

### 8.1 Inline Comments

```python
# English only, 2 spaces before #
obs = check(obs).to(**self.tpdv)  # convert numpy to tensor
```

### 8.2 Block Comments

```python
# Explain WHY, not WHAT
# Clamp importance weights to prevent excessively large policy updates
# when the new and old policies diverge significantly.
surr2 = torch.clamp(imp_weights, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
```

### 8.3 Special Markers

Mandatory for relevant situations. Enables project-wide tracking via `grep`:

```python
# TODO: add support for MultiDiscrete action spaces
# FIXME: numerical instability when advantages near zero
# HACK: temporary workaround for gym 0.26 API change
# NOTE: follows equation (7) in Kuba et al. 2022
```

### 8.4 Prohibited

```python
# BAD: states the obvious
x = x + 1  # increment x
loss = mse(pred, target)  # compute MSE loss

# BAD: mixed language in single comment
# 计算损失 compute the loss value
```

---

## 9. Migration Scope

### 9.1 Files Requiring Tab-to-Space Conversion

| Module | Files | Method |
|--------|-------|--------|
| `algorithms/twots_vvc/` | All 8 files | `expand -t 4` |
| `envs/smartgrid/base_env/` | `core_env.py`, `vvc_env.py` (partial) | Manual fix |

### 9.2 Files Requiring Type Hint Addition

| Module | Priority | Scope |
|--------|----------|-------|
| `algorithms/actors/*.py` | High | All 15 actor files |
| `algorithms/critics/*.py` | High | All 5 critic files |
| `envs/vvc/vvc_env.py` | High | Full file |
| `envs/env_wrappers.py` | High | All public methods |
| `runners/*.py` | Medium | All runner files |
| `models/**/*.py` | Medium | Public APIs |
| `common/buffers/*.py` | Medium | Public APIs |
| `utils/*.py` | Low | Public functions |

### 9.3 Environment Signature Unification

| Environment | Current Signature | Target Signature |
|-------------|------------------|-----------------|
| VVC | `(args, rank=None)` | `(args: dict[str, Any], rank: int \| None = None)` |
| SmartGrid | `(env, config, rank=None)` | `(args: dict[str, Any], rank: int \| None = None)` |
| Stackelberg | `(args: Dict[str, Any])` | `(args: dict[str, Any], rank: int \| None = None)` |
| DSR | `(args: Dict[str, Any], rank=None)` | `(args: dict[str, Any], rank: int \| None = None)` |

### 9.4 Logging Addition

All files in `algorithms/actors/`, `algorithms/critics/`, and `envs/` that currently lack
`logging.getLogger(__name__)` need it added.

---

## 10. Review Checklist

Use this checklist for all new code and during migration:

- [ ] 4-space indentation, no tabs
- [ ] `from __future__ import annotations` present
- [ ] Imports sorted: stdlib -> third-party -> project
- [ ] No relative imports
- [ ] All classes and public methods have English Google-style docstrings
- [ ] Type hints on all function signatures
- [ ] `__init__` annotated with `-> None`
- [ ] `super().__init__()` (modern style)
- [ ] `logger = logging.getLogger(__name__)` present
- [ ] `self.tpdv` used for tensor device management
- [ ] `check()` used for numpy-to-tensor conversion
- [ ] `assert` for preconditions with descriptive messages
- [ ] `try/except` only for external I/O, with `logger.error()` + re-raise
- [ ] Config required keys: `args["key"]`, optional: `args.get("key", default)`
- [ ] Comments in English, explain WHY not WHAT
- [ ] `TODO` / `FIXME` / `HACK` / `NOTE` markers used where applicable
- [ ] No silent exception swallowing
- [ ] No `camelCase` variable names
- [ ] Large number constants use `_` separator (e.g., `100_000`)
- [ ] Environment `__init__` follows unified signature

---

## Appendix A: Comparison of Old vs New Style

### Before (Old Style)

```python
"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn

from algorithms.actors.on_policy_base import OnPolicyBase
from utils.envs_tools import check
from utils.models_tools import get_grad_norm


class HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """初始化 HAPPO 算法。
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
        """
        (obs_batch, rnn_states_batch, actions_batch,
         old_action_log_probs_batch, adv_targ,
         available_actions_batch, active_masks_batch) = sample
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        ...
```

### After (New Style)

```python
# -*- coding: utf-8 -*-
"""HAPPO algorithm for heterogeneous-agent proximal policy optimization."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from algorithms.actors.on_policy_base import OnPolicyBase
from utils.envs_tools import check
from utils.models_tools import get_grad_norm

logger = logging.getLogger(__name__)


class HAPPO(OnPolicyBase):
    """HAPPO algorithm for heterogeneous-agent proximal policy optimization.

    Implements clipped surrogate objective with sequential agent updates
    and importance sampling weight propagation.

    Attributes:
        clip_param: Clipping parameter for PPO surrogate objective.
        ppo_epoch: Number of PPO training epochs per update.
    """

    def __init__(
        self,
        args: dict[str, Any],
        obs_space: gym.Space,
        act_space: gym.Space,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize HAPPO algorithm.

        Args:
            args: Algorithm configuration dictionary.
            obs_space: Observation space for the agent.
            act_space: Action space for the agent.
            device: Torch device for tensor operations.
        """
        super().__init__(args, obs_space, act_space, device)
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        logger.info(f"Initialized {self.__class__.__name__}")

    def update(self, sample: tuple) -> tuple[torch.Tensor, ...]:
        """Perform single gradient update on actor network.

        Args:
            sample: Batch data tuple containing (obs, rnn_states,
                actions, action_log_probs, advantages, masks).

        Returns:
            Tuple of (policy_loss, dist_entropy, actor_grad_norm, imp_weights).
        """
        (obs_batch, rnn_states_batch, actions_batch,
         old_action_log_probs_batch, adv_targ,
         available_actions_batch, active_masks_batch) = sample
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        ...
```

---

*This document serves as the authoritative code style reference for the PowerZoo project.
All new code and refactored code must comply with these standards.*
