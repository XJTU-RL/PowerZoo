# District Dispatch Render System - Detailed Design

## Architecture Overview

Gradio Web + Matplotlib animation hybrid. 8 Tabs, dual rendering engine (Plotly interactive + Matplotlib export).

### Design Principles
- Single Responsibility: each file ≤500 lines, one function per module
- ALL OpenDSS attributes visualized without exception
- Dual mode: real-time inference + pre-recorded playback
- Export everything: CSV, JSON, GIF, MP4, HTML

---

## Directory Structure

```
envs/district_dispatch/render/
├── render_app.py                     # Gradio main entry, Tab assembly only (~200L)
├── tabs/                             # UI layout + event binding per Tab
│   ├── tab_model_data.py             # Tab 1: Model & Data (~400L)
│   ├── tab_live_inference.py         # Tab 2: Live Inference + Manual Override (~500L)
│   ├── tab_playback.py              # Tab 3: Episode Playback (~400L)
│   ├── tab_analytics.py             # Tab 4: Analytics Dashboard (~500L)
│   ├── tab_comparison.py            # Tab 5: Multi-Episode Comparison (~400L)
│   ├── tab_stress_test.py           # Tab 6: Scenario Stress Test (~400L)
│   ├── tab_training.py              # Tab 7: Training Progress (~300L)
│   └── tab_export.py                # Tab 8: Export Center (~350L)
├── engine/                           # Backend computation
│   ├── checkpoint_loader.py          # PyTorch checkpoint parse (~250L)
│   ├── inference_engine.py           # Actor model inference (~350L)
│   ├── episode_runner.py             # Drive core.step() loop (~300L)
│   ├── episode_recorder.py           # Write step data to file (~250L)
│   ├── episode_reader.py             # Load recorded episode (~200L)
│   ├── manual_override.py            # Action override logic (~200L)
│   └── stress_test_runner.py         # Parameter sweep + batch inference (~300L)
├── data/                             # OpenDSS data extraction
│   ├── bus_data_extractor.py         # Bus: voltage/current/power/coords (~400L)
│   ├── line_data_extractor.py        # Line: current/power/loss/loading (~350L)
│   ├── device_data_extractor.py      # PV/Storage/EV/Capacitor (~400L)
│   ├── transformer_data_extractor.py # Transformer: tap/loss/current (~300L)
│   ├── circuit_data_extractor.py     # System: total loss/power/convergence (~300L)
│   ├── regulator_data_extractor.py   # RegControl: tap position/params (~250L)
│   └── snapshot_assembler.py         # Collect all extractors → unified dict (~300L)
├── viz/
│   ├── plotly/                       # Plotly interactive (Web)
│   │   ├── topology_graph.py         # Circuit topology with devices (~500L)
│   │   ├── voltage_heatmap.py        # Bus voltage distribution (~250L)
│   │   ├── power_flow_diagram.py     # Line power flow arrows (~350L)
│   │   ├── device_schedule_chart.py  # PV/Storage/EV time series (~400L)
│   │   ├── reward_breakdown.py       # Radar + stacked bar (~300L)
│   │   ├── voltage_profile.py        # Distance-voltage curve (~250L)
│   │   └── exchange_sankey.py        # Power exchange Sankey (~250L)
│   ├── mpl/                          # Matplotlib (export)
│   │   ├── topology_animator.py      # Topology frame generation (~400L)
│   │   ├── episode_animation.py      # GIF/MP4 synthesis (~350L)
│   │   └── static_report_plots.py    # Publication-grade figures (~400L)
│   └── theme.py                      # Colors/styles/constants (~150L)
├── export/
│   ├── csv_exporter.py               # All OpenDSS data to CSV (~300L)
│   ├── json_exporter.py              # Episode data to JSON (~200L)
│   ├── animation_exporter.py         # GIF/MP4 export (~250L)
│   ├── report_generator.py           # HTML report (~400L)
│   └── topology_html_exporter.py     # Standalone interactive HTML (~250L)
├── utils/
│   ├── training_log_parser.py        # Parse progress.txt/tensorboard (~250L)
│   └── color_scales.py               # Voltage/power/SOC color maps (~150L)
└── assets/
    └── bus_coordinates.py            # IEEE34 bus XY loader (~100L)
```

---

## Tab 1: Model & Data

### UI Layout
```
┌─────────────────────────────────────────────────────────────┐
│ [Checkpoint Directory]  [Browse...]  [Scan]                 │
│                                                             │
│ ┌─ Available Checkpoints ──────────────────────────────┐    │
│ │ ☑ checkpoint_episode_520  (2026-02-07 14:09)         │    │
│ │ ○ checkpoint_episode_480  (2026-02-07 14:05)         │    │
│ │ ○ checkpoint_episode_440  (2026-02-07 14:01)         │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                             │
│ [Load Model]  Status: ✓ Model loaded (3 actors + critic)    │
│                                                             │
│ ┌─ Model Info ─────────────────────────────────────────┐    │
│ │ Algorithm: HAPPO    Agents: 3    Hidden: 128         │    │
│ │ Obs Dim: [17, 17, 13]   Action Dim: [8, 8, 6]       │    │
│ │ Training Steps: 50,000   Episodes: 520               │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                             │
│ ┌─ Environment Config ─────────────────────────────────┐    │
│ │ System: District_34Bus_3Zone    Districts: 3         │    │
│ │ Episode Length: 96 steps (24h @ 15min)                │    │
│ │ [Edit Reward Weights]  [Edit Constraints]             │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                             │
│ ┌─ Recorded Episodes ─────────────────────────────────┐     │
│ │ (List of .npz files for playback tab)                │     │
│ └──────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Functionality
- Scan results directory for checkpoint folders
- Parse config.json to extract training metadata
- Load PyTorch actor/critic models into memory
- Display model architecture summary
- List available recorded episodes

### Data Flow
```
User selects directory → checkpoint_loader.py scans →
User clicks Load → inference_engine.py loads models →
State shared to Tab 2/3/5/6 via gr.State
```

---

## Tab 2: Live Inference

### UI Layout
```
┌────────────────────────────────────────────────────────────────┐
│ [▶ Start] [⏸ Pause] [⏭ Step] [⏹ Stop] [🔴 Record]  Speed: [1x▼]│
│ Step: 42/96   Time: 10:30   Price: 0.65 yuan/kWh              │
│                                                                │
│ ┌─ Circuit Topology (Plotly) ─────────────────────────────┐    │
│ │                                                         │    │
│ │   [Interactive IEEE 34-bus topology]                     │    │
│ │   - Nodes colored by voltage (green/yellow/red)         │    │
│ │   - Lines colored by loading % (blue→red)               │    │
│ │   - Device icons: ☀PV ⚡Storage 🔌EV 🔄RegCtrl         │    │
│ │   - Zone backgrounds: Zone0=blue, Zone1=green, Zone2=orange│  │
│ │   - Power flow arrows on tie-lines                      │    │
│ │   - Hover: all bus/line/device details                  │    │
│ │                                                         │    │
│ └─────────────────────────────────────────────────────────┘    │
│                                                                │
│ ┌─ Device Panel ──────────┐  ┌─ System Metrics ───────────┐   │
│ │ Zone 0:                 │  │ Total Load:  1,234 kW       │   │
│ │  PV1: 180/200kW ██████░│  │ Total Gen:     890 kW       │   │
│ │  PV2: 120/200kW ████░░░│  │ Total Loss:   45.2 kW       │   │
│ │  ESS: SOC 65% ████████░│  │ Voltage Range: 0.96-1.04 pu │   │
│ │       → Discharge 50kW │  │ Convergence: ✓ (4 iter)     │   │
│ │  EV:  30/50kW ██████░░░│  │ Exchange 0→1: +120 kW       │   │
│ │ Zone 1:                 │  │ Exchange 1→2: -30 kW        │   │
│ │  PV1: 190/200kW ████████│  │                             │   │
│ │  ...                    │  │ Loss Rate: 3.67%            │   │
│ └─────────────────────────┘  └─────────────────────────────┘   │
│                                                                │
│ ┌─ Manual Override (Toggle) ──────────────────────────────┐    │
│ │ [✓ Enable Override]   Agent: [Zone 0 ▼]                 │    │
│ │ PV Curtail 1: [====●=====] 0.85                         │    │
│ │ PV Curtail 2: [====●=====] 0.60                         │    │
│ │ Storage:      [==●=======] -0.50 (Charge)               │    │
│ │ EV Modulate:  [======●===] 0.70                         │    │
│ │ Exchange P:   [====●=====] +120 kW                      │    │
│ │ [Apply Override]  [Reset to Model]                      │    │
│ └─────────────────────────────────────────────────────────┘    │
│                                                                │
│ ┌─ Reward Decomposition (live) ──────────────────────────┐     │
│ │ Economic: -0.32  Voltage: -0.05  Loss: -0.18           │     │
│ │ Carbon: +0.12   Exchange: -0.08  Storage: -0.02        │     │
│ │ Total: -0.53                                            │     │
│ └─────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

### Functionality
- Auto-play / step-by-step / pause controls
- Real-time OpenDSS solve at each step
- Plotly topology updates with ALL attributes on hover:
  - Bus hover: Vpu (3 phases), angle, kW/kvar injection, connected devices
  - Line hover: I (3 phases), P/Q flow, losses, loading %, length
  - Device hover: PV output/curtailment, ESS SOC/power, EV load
  - Transformer hover: tap position, losses (load/no-load), currents
  - Regulator hover: tap number, Vreg target, bandwidth
  - Capacitor hover: kvar injection, switch states
- Recording toggle → saves episode data for playback
- Manual Override panel for interactive action modification
- Live reward decomposition display

### OpenDSS Attributes Displayed (per step)
All data extracted by snapshot_assembler.py:

**Bus Level (34 buses):**
- Vpu per phase (mag + angle), VLL, sequence voltages (V0, V+, V-)
- Connected PCE/PDE lists
- Coordinates (for topology positioning)

**Line Level (32 lines):**
- Current per phase (mag + angle), loading %
- Power per phase (P + Q), total power
- Losses (P + Q), phase losses
- Sequence currents/powers
- Normal/Emergency amps rating

**Transformer (SubXF + XFM1):**
- Per-winding: kV, kVA, R%, tap position
- Currents per winding
- Losses by type: total, load, no-load
- Loading %

**Regulator (6 units):**
- Tap number, tap position (pu)
- Vreg target, bandwidth
- Forward/Reverse R/X compensation

**Capacitor (C844, C848):**
- kvar output, step states
- Current, voltage at terminal

**PV Systems (5 units):**
- kW output, kvar, PF
- Pmpp, irradiance now, kVA rated
- Terminal voltage/current
- Curtailment ratio (from agent action)

**Storage (3 units):**
- SOC (pu), state (idle/charge/discharge)
- kW in/out, DC power
- All 34 internal variables (losses, efficiency, etc.)
- Terminal voltage/current

**EV Charger Loads (2 units):**
- kW demand, kvar, PF
- Current, voltage at terminal

**System Level:**
- Total losses (kW + kvar), line losses, substation losses
- Total power from source
- All bus Vmag pu (for statistics)
- Solution: converged, iterations, mode

---

## Tab 3: Episode Playback

### UI Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Episode: [recorded_ep_001.npz ▼]  [Load]                      │
│                                                                │
│ ┌─ Time Slider ──────────────────────────────────────────┐     │
│ │ |●═══════════════════════════════════════════════| 42/96│     │
│ │ 00:00                  12:00                    24:00   │     │
│ └────────────────────────────────────────────────────────┘     │
│ [◀◀] [◀] [▶ Play] [▶▶]   Speed: [1x ▼]                       │
│                                                                │
│ ┌─ Topology View ──────────────────────────────────────┐       │
│ │  (Same as Live Inference - Plotly interactive)        │       │
│ │  - All OpenDSS attributes on hover                   │       │
│ │  - Slider position drives displayed step             │       │
│ └──────────────────────────────────────────────────────┘       │
│                                                                │
│ ┌─ Timeline Charts ─────────────────────────────────────┐      │
│ │  Voltage | Power | SOC | Reward | Price               │      │
│ │  [Multi-line chart, vertical cursor at current step]  │      │
│ │  Shows full episode trajectory with current position  │      │
│ └───────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘
```

### Functionality
- Load pre-recorded .npz episode data
- Time slider for random-access to any step
- Playback with variable speed (0.5x, 1x, 2x, 5x)
- Timeline charts show full episode trajectory
- Same topology visualization as Tab 2

### Recorded Data Format (.npz)
```python
{
    "metadata": {  # Episode metadata
        "checkpoint": "checkpoint_episode_520",
        "seed": 42,
        "timestamp": "2026-02-07T14:30:00",
        "n_steps": 96,
        "n_agents": 3,
        "total_reward": [-85.3, -78.2, -92.1],
    },
    "steps": [  # List of 96 step snapshots
        {
            "step": 0,
            "time_of_day": 0.0,
            "price": 0.35,
            "actions": ndarray(3, max_action_dim),
            "obs": ndarray(3, max_obs_dim),
            "rewards": ndarray(3, 1),
            "reward_components": {
                "economic": [-0.32, -0.28, -0.15],
                "voltage": [-0.05, -0.02, -0.01],
                ...
            },
            "bus_data": {  # ALL bus attributes
                "800": {"vpu": [1.02, 1.01, 1.03], "angle": [...], ...},
                ...
            },
            "line_data": {  # ALL line attributes
                "L1": {"current_mag": [...], "power": [...], "losses": [...], "loading_pct": 45.2},
                ...
            },
            "device_data": {  # ALL device attributes
                "pv_d0_1": {"kw": 180, "kvar": 5, "pmpp": 200, "irradiance": 0.9, ...},
                "ess_d0": {"soc": 0.65, "kw_out": 50, "state": 1, ...all 34 vars...},
                "ev_d0": {"kw": 30, "kvar": 10, ...},
                ...
            },
            "transformer_data": {...},
            "regulator_data": {...},
            "capacitor_data": {...},
            "circuit_data": {
                "total_loss_kw": 45.2, "total_loss_kvar": 32.1,
                "line_loss_kw": 38.5, "substation_loss_kw": 6.7,
                "total_power_kw": -1234, "total_power_kvar": -567,
                "converged": True, "iterations": 4,
            },
        },
        ...  # 95 more steps
    ]
}
```

---

## Tab 4: Analytics Dashboard

### UI Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Source: [Live Session ▼] or [Recorded Episode ▼]               │
│                                                                │
│ ┌─ Row 1: Voltage Analysis ────────────────────────────────┐   │
│ │ ┌─ Voltage Profile ──────┐  ┌─ Voltage Heatmap ────────┐│   │
│ │ │ Distance vs Vpu curve  │  │ Bus × Time heatmap       ││   │
│ │ │ 3 phases + limits      │  │ Color: Vpu (0.9-1.1)    ││   │
│ │ │ Per zone highlight     │  │ Rows: 34 buses           ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                │
│ ┌─ Row 2: Power Analysis ──────────────────────────────────┐   │
│ │ ┌─ Power Balance ────────┐  ┌─ Exchange Sankey ─────────┐│   │
│ │ │ Stacked area chart     │  │ Zone0 ──120kW──→ Zone1   ││   │
│ │ │ Load/PV/Storage/EV/    │  │ Zone1 ──30kW───→ Zone2   ││   │
│ │ │ Exchange/Loss over time│  │ Width = power magnitude   ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                │
│ ┌─ Row 3: Device Scheduling ───────────────────────────────┐   │
│ │ ┌─ PV Output ────────────┐  ┌─ Storage SOC + Power ────┐│   │
│ │ │ Available vs Actual    │  │ SOC curves + charge/     ││   │
│ │ │ Curtailment shaded     │  │ discharge bars           ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ │ ┌─ EV Load ──────────────┐  ┌─ Regulator Taps ─────────┐│   │
│ │ │ EV demand curves       │  │ Tap positions over time  ││   │
│ │ │ Modulation ratio       │  │ 6 regulators             ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                │
│ ┌─ Row 4: Reward & Loss ───────────────────────────────────┐   │
│ │ ┌─ Reward Breakdown ─────┐  ┌─ Loss Distribution ──────┐│   │
│ │ │ 6-component stacked    │  │ Line/Transformer/NoLoad  ││   │
│ │ │ Per agent, over time   │  │ losses pie + time series ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                │
│ ┌─ Row 5: Line Loading ────────────────────────────────────┐   │
│ │ ┌─ Loading Heatmap ──────┐  ┌─ Capacitor Status ───────┐│   │
│ │ │ Line × Time heatmap    │  │ kvar injection + states  ││   │
│ │ │ Color: loading %       │  │ Voltage at terminal      ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Charts (14 total)
1. **Voltage Profile**: Distance vs Vpu, 3 phases, with min/max limits
2. **Voltage Heatmap**: Bus × Time, color = Vpu
3. **Power Balance**: Stacked area (Load, PV, Storage, EV, Exchange, Loss)
4. **Exchange Sankey**: Inter-zone power flow diagram
5. **PV Output**: Available vs Actual, curtailment area
6. **Storage SOC + Power**: Dual-axis (SOC line + charge/discharge bars)
7. **EV Load**: Demand curves with modulation ratio
8. **Regulator Taps**: 6 regulators tap position over time
9. **Reward Breakdown**: 6-component stacked per agent
10. **Loss Distribution**: Pie (line/transformer/no-load) + time series
11. **Line Loading Heatmap**: Line × Time, color = loading %
12. **Capacitor Status**: kvar + switch states over time
13. **Sequence Voltages**: V+, V-, V0 per bus (for unbalance analysis)
14. **Current Distribution**: Phase currents on critical lines

---

## Tab 5: Multi-Episode Comparison

### UI Layout
```
┌────────────────────────────────────────────────────────────────┐
│ ┌─ Episode Selection (up to 3) ───────────────────────────┐    │
│ │ Episode A: [checkpoint_520_seed42.npz ▼]  Color: Blue   │    │
│ │ Episode B: [mappo_checkpoint_500.npz ▼]   Color: Orange │    │
│ │ Episode C: [                          ▼]  Color: Green  │    │
│ │ [Compare]                                               │    │
│ └─────────────────────────────────────────────────────────┘    │
│                                                                │
│ ┌─ Comparison Charts ─────────────────────────────────────┐    │
│ │ ┌─ Reward Comparison ────┐  ┌─ Voltage Comparison ─────┐│   │
│ │ │ Cumulative reward      │  │ Min voltage over time    ││   │
│ │ │ A/B/C overlaid         │  │ A/B/C overlaid           ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ │ ┌─ Action Comparison ────┐  ┌─ Loss Comparison ────────┐│   │
│ │ │ Per-agent action dist  │  │ Total loss over time     ││   │
│ │ │ Violin/Box plots       │  │ A/B/C overlaid           ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ └─────────────────────────────────────────────────────────┘    │
│                                                                │
│ ┌─ Summary Table ──────────────────────────────────────────┐   │
│ │ Metric           │ Episode A │ Episode B │ Episode C     │   │
│ │ Total Reward      │  -85.3   │  -92.1   │    -          │   │
│ │ Avg Voltage Min   │  0.962   │  0.955   │    -          │   │
│ │ Total Loss (kWh)  │  42.3    │  48.7    │    -          │   │
│ │ PV Utilization %  │  87.2%   │  82.1%   │    -          │   │
│ │ Voltage Violation %│  2.1%   │  5.3%    │    -          │   │
│ │ Avg SOC Range     │ 0.3-0.7  │ 0.2-0.8  │    -          │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Comparison Metrics
- Cumulative reward (per agent + total)
- Voltage compliance rate (% steps within limits)
- Network losses (kWh over episode)
- PV utilization rate (actual/available)
- Storage cycling depth
- Exchange volume
- Action distribution (violin plots)

---

## Tab 6: Scenario Stress Test

### UI Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Base Model: [checkpoint_episode_520]                           │
│                                                                │
│ ┌─ Scenario Parameters ───────────────────────────────────┐    │
│ │ Load Multiplier:    [====●=====] 1.5x  (0.5-2.0)       │    │
│ │ PV Output Ratio:    [======●===] 0.3   (0.0-1.0)       │    │
│ │ Initial SOC:        [===●======] 0.3   (0.1-0.9)       │    │
│ │ EV Demand Mult:     [=======●==] 1.8   (0.0-3.0)       │    │
│ │ Carbon Intensity:   [====●=====] 0.8   (0.0-1.2)       │    │
│ │ Price Multiplier:   [====●=====] 1.2   (0.5-2.0)       │    │
│ │                                                         │    │
│ │ [Run Single Scenario]  [Run Parameter Sweep]            │    │
│ └─────────────────────────────────────────────────────────┘    │
│                                                                │
│ ┌─ Results ────────────────────────────────────────────────┐   │
│ │  (Same topology + charts as Tab 2)                       │   │
│ │  With baseline comparison overlaid                       │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                │
│ ┌─ Sweep Results (if parameter sweep) ─────────────────────┐   │
│ │  Heatmap: Parameter1 × Parameter2 → Reward/Voltage/Loss  │   │
│ │  Sensitivity analysis bar chart                           │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Stress Test Capabilities
- Single scenario: modify parameters and run one episode
- Parameter sweep: grid search over 2 parameters, run N episodes
- Sensitivity analysis: one-at-a-time parameter variation
- Compare stressed results vs baseline

---

## Tab 7: Training Progress

### UI Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Results Dir: [/results/district_dispatch/...]  [Scan]          │
│                                                                │
│ ┌─ Training Curves ───────────────────────────────────────┐    │
│ │ ┌─ Episode Reward ───────┐  ┌─ Policy Loss ────────────┐│   │
│ │ │ Per-agent + average    │  │ Actor loss curve         ││   │
│ │ │ Smoothed + raw         │  │ Critic loss curve        ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ │ ┌─ Eval Reward ──────────┐  ┌─ Learning Rate ─────────┐│   │
│ │ │ Evaluation rewards     │  │ LR schedule curve        ││   │
│ │ │ With confidence band   │  │                          ││   │
│ │ └────────────────────────┘  └──────────────────────────┘│   │
│ └─────────────────────────────────────────────────────────┘    │
│                                                                │
│ ┌─ Training Summary Table ─────────────────────────────────┐   │
│ │ Best Reward: -76.6 (ep 512)   FPS: 39                   │   │
│ │ Total Time: 21 min            Episodes: 520              │   │
│ │ Convergence Trend: ↑ Improving                           │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

---

## Tab 8: Export Center

### UI Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Source: [Live Session ▼] or [Recorded Episode ▼]               │
│                                                                │
│ ┌─ CSV Export ─────────────────────────────────────────────┐   │
│ │ ☑ Bus Voltages (all phases, all steps)                   │   │
│ │ ☑ Line Currents & Powers (all phases, all steps)         │   │
│ │ ☑ Line Losses (per phase)                                │   │
│ │ ☑ Line Loading %                                         │   │
│ │ ☑ Transformer Data (tap, losses, currents)               │   │
│ │ ☑ PV Output (kW, kvar, Pmpp, irradiance)                 │   │
│ │ ☑ Storage State (SOC + all 34 internal variables)        │   │
│ │ ☑ EV Charger Load                                        │   │
│ │ ☑ Capacitor States                                       │   │
│ │ ☑ Regulator Tap Positions                                │   │
│ │ ☑ System Totals (loss, power, generation)                │   │
│ │ ☑ Agent Actions (raw + decoded)                          │   │
│ │ ☑ Agent Observations                                     │   │
│ │ ☑ Reward Components (6 components × 3 agents)            │   │
│ │ ☑ Sequence Voltages/Currents                             │   │
│ │ ☑ Solution Convergence Info                              │   │
│ │ [Export Selected CSVs]  → downloads zip                   │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                │
│ ┌─ Animation Export ───────────────────────────────────────┐   │
│ │ Format: [GIF ▼] / [MP4]   FPS: [4 ▼]   DPI: [150 ▼]    │   │
│ │ Content: ☑ Topology  ☑ Device Charts  ☑ Voltage Profile  │   │
│ │ [Generate Animation]  Progress: ████████░░ 80%           │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                │
│ ┌─ Report Export ──────────────────────────────────────────┐   │
│ │ [Generate HTML Report]  — Full episode analysis report   │   │
│ │ [Export Interactive Topology HTML]  — Standalone viewer   │   │
│ │ [Export Episode Data (.npz)]  — For playback tab          │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Export Formats
| Format | Content | Size Estimate |
|--------|---------|---------------|
| CSV (zip) | All OpenDSS data, 96 steps × all elements | ~5-10 MB |
| GIF | Topology animation, 96 frames | ~15-30 MB |
| MP4 | Topology animation, 96 frames | ~5-10 MB |
| HTML Report | Full analysis with embedded Plotly | ~3-5 MB |
| Interactive HTML | Standalone topology viewer | ~1-2 MB |
| NPZ | Raw episode data for replay | ~2-5 MB |
| JSON | Human-readable episode summary | ~1-3 MB |

---

## Implementation Phases

### Phase R1: Foundation (engine + data + assets)
- `assets/bus_coordinates.py`
- `utils/color_scales.py`, `utils/training_log_parser.py`
- `viz/theme.py`
- `engine/checkpoint_loader.py`
- `engine/inference_engine.py`
- `data/*_extractor.py` (all 7 files)
- `data/snapshot_assembler.py`

### Phase R2: Core Visualization
- `viz/plotly/topology_graph.py`
- `viz/plotly/voltage_heatmap.py`
- `viz/plotly/power_flow_diagram.py`
- `viz/plotly/device_schedule_chart.py`
- `viz/plotly/reward_breakdown.py`
- `viz/plotly/voltage_profile.py`
- `viz/plotly/exchange_sankey.py`

### Phase R3: Engine + Recording
- `engine/episode_runner.py`
- `engine/episode_recorder.py`
- `engine/episode_reader.py`
- `engine/manual_override.py`
- `engine/stress_test_runner.py`

### Phase R4: Matplotlib Export
- `viz/mpl/topology_animator.py`
- `viz/mpl/episode_animation.py`
- `viz/mpl/static_report_plots.py`

### Phase R5: Export System
- `export/csv_exporter.py`
- `export/json_exporter.py`
- `export/animation_exporter.py`
- `export/report_generator.py`
- `export/topology_html_exporter.py`

### Phase R6: Gradio Tabs
- `tabs/tab_model_data.py`
- `tabs/tab_live_inference.py`
- `tabs/tab_playback.py`
- `tabs/tab_analytics.py`
- `tabs/tab_comparison.py`
- `tabs/tab_stress_test.py`
- `tabs/tab_training.py`
- `tabs/tab_export.py`
- `render_app.py`

### Phase R7: Integration Testing
- End-to-end test with real checkpoint
- All tabs functional verification
