# Long Methods Refactoring Plan

> Generated: 2025-11-26
> Total long methods (>50 lines): 226
> Total lines in long methods: ~18,642

## Summary Statistics

| Directory | Count | Total Lines | Avg Lines | Priority |
|-----------|-------|-------------|-----------|----------|
| runners | 24 | 3,130 | 130.4 | HIGH |
| common | 21 | 2,023 | 96.3 | MEDIUM |
| algorithms | 28 | 2,154 | 76.9 | MEDIUM |
| envs | 128 | 9,738 | 76.1 | MEDIUM |
| utils | 18 | 1,191 | 66.2 | LOW |
| models | 7 | 406 | 58.0 | LOW |

---

## Priority 1: Critical (>150 lines) - 15 methods

### 1. runners/Qmix_runner.py::separated_collect_rollout
- **Lines**: 218-560 (343 lines)
- **Purpose**: Collects rollout data for QMIX algorithm
- **Recommendation**: Split into `_collect_observations()`, `_execute_actions()`, `_process_rewards()`, `_update_buffers()`

### 2. envs/powerzoo_llm/base_env/env.py::step
- **Lines**: 370-662 (293 lines)
- **Purpose**: Environment step function
- **Recommendation**: Extract `_process_actions()`, `_run_power_flow()`, `_compute_rewards()`, `_build_observations()`

### 3. runners/Qmix_base_runner.py::__init__
- **Lines**: 54-299 (246 lines)
- **Purpose**: Runner initialization
- **Recommendation**: Extract `_init_buffers()`, `_init_algorithms()`, `_init_logger()`, `_load_pretrained()`

### 4. common/buffers/shared_on_policy_actor_buffer.py::recurrent_generator
- **Lines**: 561-799 (239 lines)
- **Purpose**: Generates recurrent data batches
- **Recommendation**: Split into `_prepare_recurrent_data()`, `_generate_mini_batches()`, `_shuffle_indices()`

### 5. runners/off_policy_ha_runner.py::train
- **Lines**: 32-260 (229 lines)
- **Purpose**: Main training loop for heterogeneous agents
- **Recommendation**: Extract `_train_epoch()`, `_evaluate()`, `_log_metrics()`

### 6. runners/on_policy_base_runner.py::__init__
- **Lines**: 45-260 (216 lines)
- **Purpose**: Base runner initialization
- **Recommendation**: Extract `_setup_environments()`, `_setup_algorithms()`, `_setup_buffers()`

### 7. runners/off_policy_base_runner.py::__init__
- **Lines**: 59-240 (182 lines)
- **Purpose**: Off-policy runner initialization
- **Recommendation**: Extract initialization sub-methods

### 8. algorithms/critics/soft_twin_continuous_q_critic.py::train
- **Lines**: 87-260 (174 lines)
- **Purpose**: SAC critic training
- **Recommendation**: Split into `_compute_loss()`, `_update_targets()`, `_log_diagnostics()`

### 9. envs/powerzoo_llm/logging/powerzoo_llm_logger.py::per_step
- **Lines**: 214-380 (167 lines)
- **Purpose**: Per-step logging
- **Recommendation**: Extract metric collection methods

### 10. envs/powerzoo/powerzoo_logger.py::per_step
- **Lines**: 120-285 (166 lines)
- **Purpose**: Per-step logging
- **Recommendation**: Same as above

### 11. algorithms/actors/m_Qmix.py::train_policy_on_batch
- **Lines**: 92-255 (164 lines)
- **Purpose**: QMIX policy training
- **Recommendation**: Split computation and update phases

### 12. algorithms/actors/hatrpo.py::update
- **Lines**: 57-214 (158 lines)
- **Purpose**: HATRPO policy update
- **Recommendation**: Extract `_compute_advantages()`, `_compute_kl()`, `_line_search()`

### 13. runners/on_policy_base_runner.py::_collect_heterogeneous
- **Lines**: 453-599 (147 lines)
- **Purpose**: Heterogeneous data collection
- **Recommendation**: Modularize by agent type

### 14. utils/sign.py::add_header_to_file
- **Lines**: 21-167 (147 lines)
- **Purpose**: Add header to source files
- **Recommendation**: Extract template generation and file processing

### 15. common/buffers/shared_on_policy_actor_buffer.py::feed_forward_generator
- **Lines**: 417-559 (143 lines)
- **Purpose**: Feed-forward batch generation
- **Recommendation**: Similar to recurrent_generator

---

## Priority 2: High (100-150 lines) - 23 methods

| File | Method | Lines | Line Count |
|------|--------|-------|------------|
| envs/dsr/dsr_env.py | _build_agent_observation | 463-604 | 142 |
| runners/on_policy_ha_runner.py | train | 108-249 | 142 |
| runners/on_policy_ha_runner.py | _log_training_metrics | 317-458 | 142 |
| envs/powerzoo_llm/logging/powerzoo_llm_logger.py | _log_episode_metrics | 589-727 | 139 |
| envs/powerzoo/powerzoo_logger.py | episode_log | 288-425 | 138 |
| runners/two_ts_runner.py | train | 74-210 | 137 |
| envs/dsr/core/dsr_core.py | _initialize_loads_per_agent | 287-418 | 132 |
| algorithms/actors/sn_mappo.py | update_leader | 111-239 | 129 |
| envs/powerzoo_llm/base_env/powerzoo_env.py | step | 187-314 | 128 |
| envs/powerzoo_llm/rewards/powerzoo_reward.py | calculate_total_reward | 89-213 | 125 |
| algorithms/actors/hasac.py | train | 96-218 | 123 |
| envs/powerzoo/powerzoo/env.py | reset | 267-388 | 122 |
| runners/Qmix_base_runner.py | run | 301-420 | 120 |
| common/buffers/on_policy_actor_buffer.py | recurrent_generator | 319-436 | 118 |
| runners/on_policy_ma_runner.py | train | 26-141 | 116 |
| envs/powerzoo/powerzoo/env.py | step | 118-231 | 114 |
| algorithms/twots_vvc/coordinator.py | train_step | 122-234 | 113 |
| envs/dsr/dsr_env.py | step | 171-282 | 112 |
| runners/off_policy_ma_runner.py | train | 24-133 | 110 |
| envs/powerzoo_llm/data_process/loadprofile_core.py | __init__ | 47-153 | 107 |
| algorithms/actors/sn_mappo.py | update_follower | 263-366 | 104 |
| envs/dsr/core/dsr_core.py | reset | 96-197 | 102 |
| common/buffers/on_policy_actor_buffer.py | feed_forward_generator | 210-308 | 99 |

---

## Priority 3: Medium (75-100 lines) - 42 methods

<details>
<summary>Click to expand</summary>

| File | Method | Lines | Line Count |
|------|--------|-------|------------|
| algorithms/actors/happo.py | train | 140-231 | 92 |
| algorithms/actors/mappo.py | train | 103-193 | 91 |
| envs/powerzoo_llm/circuit_system/circuit.py | initialize | 145-234 | 90 |
| runners/on_policy_base_runner.py | run | 262-350 | 89 |
| envs/dsr/dsr_env.py | reset | 97-183 | 87 |
| algorithms/twots_vvc/slow_sacd.py | update | 89-174 | 86 |
| envs/powerzoo_llm/base_env/env.py | reset | 213-297 | 85 |
| algorithms/actors/haa2c.py | train | 82-165 | 84 |
| runners/off_policy_base_runner.py | run | 242-324 | 83 |
| envs/stackelberg/stackelberg_game/stackelberg_base_env.py | step | 167-248 | 82 |
| ... (32 more methods) | | | 75-82 |

</details>

---

## Priority 4: Low (50-75 lines) - 146 methods

These are borderline cases. Consider refactoring only if:
- They contain deeply nested logic
- They have multiple responsibilities
- They are frequently modified

---

## Refactoring Guidelines

### Strategy 1: Extract Method
For methods with distinct phases:
```python
# Before
def train(self):
    # Phase 1: data preparation (30 lines)
    # Phase 2: forward pass (40 lines)
    # Phase 3: backward pass (30 lines)
    # Phase 4: logging (20 lines)

# After
def train(self):
    data = self._prepare_data()
    output = self._forward_pass(data)
    loss = self._backward_pass(output)
    self._log_metrics(loss)
```

### Strategy 2: Extract Class
For methods with shared state:
```python
# Extract logging logic to separate class
class TrainingLogger:
    def log_step(self, metrics): ...
    def log_episode(self, metrics): ...
    def log_eval(self, metrics): ...
```

### Strategy 3: Use Composition
For environment step methods:
```python
class PowerZooEnv:
    def __init__(self):
        self.action_processor = ActionProcessor()
        self.reward_calculator = RewardCalculator()
        self.observation_builder = ObservationBuilder()

    def step(self, actions):
        processed = self.action_processor.process(actions)
        reward = self.reward_calculator.compute(processed)
        obs = self.observation_builder.build()
        return obs, reward, done, info
```

---

## Recommended Refactoring Order

1. **Week 1**: runners/ directory (highest avg line count)
   - Focus on `__init__` methods and `train` methods

2. **Week 2**: common/buffers/ directory
   - Focus on generator methods

3. **Week 3**: algorithms/ directory
   - Focus on `train` and `update` methods

4. **Week 4**: envs/ logging methods
   - Extract to shared logging utility

---

## Notes

- Total estimated refactoring time: 4-6 weeks for critical methods
- Each refactoring should include corresponding test updates
- Maintain backward compatibility during refactoring
- Use incremental commits for each method refactored
