# DAN-HAPPO Algorithm Implementation

## Overview

This implementation integrates the Dynamic Agent Network (DAN) architecture with the Heterogeneous-Agent Proximal Policy Optimization (HAPPO) algorithm to address the limitations identified in the original DSR environment and algorithm.

## Key Features

### 1. Dynamic Agent Network (DAN) Architecture

- **Dual Encoders**: Separate encoding for environmental and agent interaction information
- **Attention Mechanism**: Multi-head attention for aggregating neighboring agent features
- **Dynamic Topology Support**: Handles varying numbers of agents and observation dimensions
- **Feature Fusion**: Combines environmental and agent features for enhanced coordination

### 2. Optimized DSR Environment

- **Improved Reward Function**:
  - Increased overload penalty weight from 1.0 to 5.0
  - Added severe overload penalty (20.0x weight)
  - Progressive overload penalties based on severity levels
  - Severe overload threshold at 1.5x normal capacity

- **Enhanced Action Masking**:
  - Safety-based action filtering
  - Conservative masking during severe overloads
  - Voltage violation prevention
  - Power flow impact prediction

- **Termination Conditions**:
  - Early termination on severe overloads
  - Optional voltage violation termination
  - System stability monitoring

### 3. DAN-HAPPO Algorithm

- **Neighbor Observation Processing**: Collects and processes observations from neighboring agents
- **Attention-based Coordination**: Uses attention weights to focus on relevant neighbors
- **Enhanced Policy Network**: Integrates DAN-encoded features into HAPPO policy
- **Specialized Training**: Custom training loop for DAN-specific features

## File Structure

```
PowerZoo/
├── models/base/
│   └── dan.py                    # DAN architecture implementation
├── algorithms/actors/
│   └── dan_happo.py             # DAN-HAPPO algorithm
├── envs/dsr/
│   └── dsr_env_optimized.py     # Optimized DSR environment
├── runner/shared/
│   └── dsr_dan_runner.py        # DAN-specific training runner
├── utils/
│   └── dan_buffer.py            # Extended replay buffer for DAN
├── configs/
│   └── dan_happo_config.py      # Configuration parameters
└── train_dan_happo.py           # Training script
```

## Configuration Parameters

### DAN Architecture
- `dan_hidden_dim`: Hidden dimension for DAN (default: 128)
- `dan_num_heads`: Number of attention heads (default: 4)
- `dan_dropout`: Dropout rate (default: 0.1)
- `use_layer_norm`: Whether to use layer normalization (default: True)
- `env_obs_ratio`: Ratio of environmental observations (default: 0.7)
- `max_neighbors`: Maximum number of neighbors (default: 5)

### Enhanced Environment
- `reward_overload`: Overload penalty weight (default: 5.0)
- `reward_severe_overload`: Severe overload penalty weight (default: 20.0)
- `severe_overload_threshold`: Severe overload threshold (default: 1.5)
- `progressive_overload_penalty`: Enable progressive penalties (default: True)
- `use_enhanced_action_mask`: Enable enhanced action masking (default: True)

### Training Parameters
- `dan_lr`: Learning rate for DAN (default: 3e-4)
- `dan_weight_decay`: Weight decay for DAN (default: 1e-5)
- `dan_grad_clip`: Gradient clipping for DAN (default: 1.0)

## Usage

### Training

```bash
python train_dan_happo.py \
    --env_name DSR \
    --algorithm_name dan_happo \
    --experiment_name test_run \
    --case_path ./envs/cases/13Bus/IEEE13Nodeckt.dss \
    --num_agents 4 \
    --episode_length 200 \
    --n_rollout_threads 8 \
    --use_dan \
    --use_neighbor_obs \
    --use_enhanced_action_mask \
    --progressive_overload_penalty
```

### Key Training Arguments

- `--use_dan`: Enable DAN architecture
- `--use_neighbor_obs`: Enable neighbor observation collection
- `--use_enhanced_action_mask`: Enable enhanced action masking
- `--progressive_overload_penalty`: Enable progressive overload penalties
- `--terminate_on_severe_overload`: Terminate episodes on severe overloads

## Algorithm Improvements

### 1. Reward Function Optimization

**Original Issues**:
- Overload penalty weight (1.0) too small compared to restoration reward (20.0)
- No progressive penalty mechanism
- Lack of severe overload handling

**Improvements**:
- Increased overload penalty to 5.0x
- Added severe overload penalty at 20.0x
- Implemented progressive penalties: [1.0, 2.0, 5.0, 10.0] for different severity levels
- Added exponential scaling for extreme overloads

### 2. Action Masking Enhancement

**Original Issues**:
- Basic action masking insufficient for safety
- No prediction of action consequences
- Limited consideration of system state

**Improvements**:
- Enhanced action masking based on safety predictions
- Conservative masking during severe overloads
- Voltage violation prevention
- Power flow impact estimation

### 3. Dynamic Network Support

**Original Issues**:
- Fixed network topology assumptions
- Limited scalability to different agent numbers
- Poor coordination between agents

**Improvements**:
- DAN architecture handles dynamic topologies
- Attention mechanism for neighbor coordination
- Scalable to varying agent numbers
- Enhanced information sharing between agents

## Monitoring and Analysis

### DAN-Specific Metrics

- **Attention Entropy**: Measures attention distribution across neighbors
- **Neighbor Count**: Average number of active neighbors
- **Action Mask Ratio**: Proportion of available actions
- **Attention Weights**: Distribution of attention across neighbors

### Training Logs

The training process logs additional DAN-specific metrics:

```
DAN Metrics:
  Attention Entropy: 1.2345
  Avg Neighbor Count: 3.45
  Action Mask Ratio: 0.7890
```

## Performance Expectations

### Improvements Over Original Implementation

1. **Safety**: Reduced constraint violations through enhanced penalties and action masking
2. **Coordination**: Better agent coordination through attention mechanism
3. **Scalability**: Support for dynamic network topologies and agent numbers
4. **Stability**: More stable training through improved reward shaping
5. **Efficiency**: Faster convergence through better information sharing

### Expected Training Behavior

- **Early Training**: High attention entropy as agents explore coordination strategies
- **Mid Training**: Decreasing overload violations as safety constraints are learned
- **Late Training**: Stable attention patterns and consistent safety compliance

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce `max_neighbors` or `dan_hidden_dim`
2. **Slow Training**: Increase `n_rollout_threads` or reduce `episode_length`
3. **Unstable Training**: Reduce `dan_lr` or increase `dan_grad_clip`
4. **Poor Coordination**: Increase `dan_num_heads` or adjust `env_obs_ratio`

### Debug Mode

Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Hierarchical Attention**: Multi-level attention for complex topologies
2. **Adaptive Penalties**: Dynamic penalty adjustment based on training progress
3. **Predictive Masking**: More sophisticated action consequence prediction
4. **Transfer Learning**: Pre-trained DAN models for different network topologies
5. **Multi-Objective Optimization**: Explicit handling of multiple objectives

## References

1. Original HAPPO paper: [Heterogeneous-Agent Proximal Policy Optimization](https://arxiv.org/abs/2109.11251)
2. Attention mechanism: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. Multi-agent coordination: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
