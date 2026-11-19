# -*- coding: utf-8 -*-
'''vvc_benchmark_metrics 的单元测试（pytest，离线、确定性）。

聚焦：按每步「实际步长区间」的能量积分，以及完整 / 截断 / 求解失败 /
非法输入四类路径，另含本地 JSONL 命令行入口测试。无网络、无随机。
'''

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vvc_benchmark_metrics import main, summarize_episode  # noqa: E402


RESULT_KEYS = {
    'valid',
    'eligible_for_day_ranking',
    'day_ranking_refusal',
    'voltage_limits_pu',
    'voltage_violation_interval_hours',
    'diagnostic_voltage_violation_interval_hours',
    'voltage_max_exceedance_pu',
    'diagnostic_voltage_max_exceedance_pu',
    'status',
    'expected_steps',
    'steps_completed',
    'steps_processed',
    'energy_kwh',
    'diagnostic_energy_kwh',
    'voltage_min_pu',
    'voltage_max_pu',
    'voltage_violation_count_total',
    'step_seconds_min',
    'step_seconds_mean',
    'step_seconds_max',
    'failure',
}

REQUIRED_METRIC_FIELDS = (
    'step_seconds',
    'power_loss_kw',
    'voltage_violation_count',
    'voltage_min_pu',
    'voltage_max_pu',
)


def make_record(**overrides):
    record = {
        'dss_converged': True,
        'step_seconds': 60.0,
        'power_loss_kw': 120.0,
        'voltage_violation_count': 0,
        'voltage_min_pu': 0.98,
        'voltage_max_pu': 1.03,
    }
    record.update(overrides)
    return record


def write_jsonl(path, payloads):
    lines = [json.dumps(payload, ensure_ascii=False) for payload in payloads]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------- 完整回合：按实际（可变）步长区间积分 ----------


def test_complete_episode_integrates_actual_varying_intervals():
    records = [
        make_record(),  # 60 秒 / 120 kW / 0 越限 / 0.98–1.03 p.u.
        make_record(step_seconds=90.0, power_loss_kw=60.0, voltage_violation_count=2, voltage_min_pu=0.95, voltage_max_pu=1.06),
        make_record(step_seconds=30.0, power_loss_kw=240.0, voltage_violation_count=1, voltage_min_pu=0.99, voltage_max_pu=1.02),
    ]
    result = summarize_episode(records, 3)
    assert result['valid'] is True
    assert result['status'] == 'valid'
    assert result['failure'] is None
    assert result['steps_completed'] == 3
    assert result['steps_processed'] == 3
    # 逐步实际区间积分：(120×60 + 60×90 + 240×30) / 3600 = 5.5 kWh
    assert result['energy_kwh'] == pytest.approx(5.5)
    assert result['diagnostic_energy_kwh'] == pytest.approx(5.5)
    assert result['voltage_min_pu'] == pytest.approx(0.95)
    assert result['voltage_max_pu'] == pytest.approx(1.06)
    assert result['voltage_violation_count_total'] == 3
    assert result['step_seconds_min'] == pytest.approx(30.0)
    assert result['step_seconds_mean'] == pytest.approx(60.0)
    assert result['step_seconds_max'] == pytest.approx(90.0)


def test_result_schema_is_explicit():
    result = summarize_episode([make_record()], 1)
    assert set(result) == RESULT_KEYS


def test_accepts_any_iterable_of_records():
    result = summarize_episode(iter([make_record(), make_record()]), 2)
    assert result['valid'] is True


# ---------- 截断（不完整）回合 ----------


def test_truncated_episode_reports_partial_diagnostic_only():
    records = [
        make_record(),
        make_record(step_seconds=90.0, power_loss_kw=60.0, voltage_violation_count=2, voltage_min_pu=0.95, voltage_max_pu=1.06),
    ]
    result = summarize_episode(records, 4)
    assert result['status'] == 'incomplete'
    assert result['valid'] is False
    assert result['energy_kwh'] is None
    assert result['diagnostic_energy_kwh'] == pytest.approx(3.5)
    assert result['steps_completed'] == 2
    assert result['steps_processed'] == 2
    assert result['failure'] is None
    assert result['voltage_violation_count_total'] == 2
    assert result['voltage_min_pu'] == pytest.approx(0.95)
    assert result['voltage_max_pu'] == pytest.approx(1.06)


def test_empty_records_is_incomplete_not_error():
    result = summarize_episode([], 3)
    assert result['status'] == 'incomplete'
    assert result['valid'] is False
    assert result['energy_kwh'] is None
    assert result['diagnostic_energy_kwh'] == 0.0
    assert result['steps_completed'] == 0
    assert result['voltage_min_pu'] is None
    assert result['step_seconds_mean'] is None
    assert result['failure'] is None


# ---------- 求解失败 ----------


def test_solver_failure_flags_stops_and_excludes_failed_step():
    records = [
        make_record(),
        make_record(step_seconds=90.0, power_loss_kw=60.0, voltage_violation_count=2, voltage_min_pu=0.95, voltage_max_pu=1.06),
        {
            'dss_converged': False,
            'power_loss_kw': -999.0,
        },  # 失败步：其余字段允许缺省或无效，一律不读取
        make_record(step_seconds=float('nan'), power_loss_kw=float('inf')),  # 失败之后的记录不被消费
    ]
    result = summarize_episode(records, 4)
    assert result['status'] == 'solver_failure'
    assert result['valid'] is False
    assert result['energy_kwh'] is None
    assert result['diagnostic_energy_kwh'] == pytest.approx(3.5)
    assert result['steps_completed'] == 2
    assert result['steps_processed'] == 3
    assert result['failure'] == {
        'step_index': 2,
        'reason': 'dss_converged=false',
        'trailing_records_ignored': 1,
    }
    assert result['voltage_min_pu'] == pytest.approx(0.95)
    assert result['voltage_violation_count_total'] == 2


def test_failure_step_with_valid_physics_is_still_excluded():
    result = summarize_episode([make_record(), make_record(dss_converged=False)], 2)
    assert result['diagnostic_energy_kwh'] == pytest.approx(2.0)
    assert result['steps_completed'] == 1
    assert result['steps_processed'] == 2
    assert result['failure']['step_index'] == 1
    assert result['failure']['trailing_records_ignored'] == 0


def test_failure_at_first_step():
    result = summarize_episode([{'dss_converged': False}], 2)
    assert result['status'] == 'solver_failure'
    assert result['valid'] is False
    assert result['energy_kwh'] is None
    assert result['diagnostic_energy_kwh'] == 0.0
    assert result['voltage_min_pu'] is None
    assert result['voltage_max_pu'] is None
    assert result['step_seconds_min'] is None
    assert result['step_seconds_mean'] is None
    assert result['step_seconds_max'] is None
    assert result['voltage_violation_count_total'] == 0
    assert result['steps_completed'] == 0
    assert result['steps_processed'] == 1


# ---------- 非法输入 ----------


def test_input_longer_than_expected_rejected():
    records = [make_record() for _ in range(3)]
    with pytest.raises(ValueError, match='超过'):
        summarize_episode(records, 2)


@pytest.mark.parametrize('bad', [0, -3, True, 2.5, '4', None])
def test_expected_steps_validated(bad):
    with pytest.raises(ValueError, match='正整数'):
        summarize_episode([make_record()], bad)


@pytest.mark.parametrize('value', [1, 0, 'true', None, 1.0])
def test_dss_converged_must_be_strict_bool(value):
    with pytest.raises(ValueError, match='布尔'):
        summarize_episode([make_record(dss_converged=value)], 1)


def test_dss_converged_required():
    record = make_record()
    del record['dss_converged']
    with pytest.raises(ValueError, match='缺少字段'):
        summarize_episode([record], 1)


@pytest.mark.parametrize('field', REQUIRED_METRIC_FIELDS)
def test_missing_required_field_rejected(field):
    record = make_record()
    del record[field]
    with pytest.raises(ValueError, match='缺少字段'):
        summarize_episode([record], 1)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('step_seconds', 0.0),
        ('step_seconds', -60.0),
        ('step_seconds', float('nan')),
        ('step_seconds', float('inf')),
        ('power_loss_kw', -1.0),
        ('power_loss_kw', float('nan')),
        ('voltage_violation_count', -1),
        ('voltage_violation_count', 1.5),
        ('voltage_violation_count', True),
        ('voltage_violation_count', '3'),
        ('voltage_min_pu', float('nan')),
        ('voltage_max_pu', float('inf')),
    ],
)
def test_invalid_field_value_rejected(field, value):
    with pytest.raises(ValueError):
        summarize_episode([make_record(**{field: value})], 1)


def test_voltage_bounds_order_validated():
    with pytest.raises(ValueError, match='voltage_min_pu'):
        summarize_episode([make_record(voltage_min_pu=1.05, voltage_max_pu=0.95)], 1)


def test_record_must_be_mapping():
    with pytest.raises(ValueError, match='映射'):
        summarize_episode([make_record(), [1, 2]], 2)


def test_records_must_be_iterable():
    with pytest.raises(ValueError, match='可迭代'):
        summarize_episode(7, 1)


# ---------- 本地 JSONL 命令行入口 ----------


def test_cli_summarizes_jsonl_file(tmp_path, capsys):
    payloads = [
        make_record(),
        make_record(step_seconds=90.0, power_loss_kw=60.0, voltage_violation_count=2, voltage_min_pu=0.95, voltage_max_pu=1.06),
        make_record(step_seconds=30.0, power_loss_kw=240.0, voltage_violation_count=1, voltage_min_pu=0.99, voltage_max_pu=1.02),
    ]
    data_file = tmp_path / 'episode.jsonl'
    write_jsonl(data_file, payloads)
    exit_code = main([str(data_file), '--expected-steps', '3'])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ''
    assert '\n' not in captured.out.strip()
    result = json.loads(captured.out)
    assert result['valid'] is True
    assert result['status'] == 'valid'
    assert result['energy_kwh'] == pytest.approx(5.5)
    assert result['diagnostic_energy_kwh'] == pytest.approx(5.5)
    assert result['voltage_violation_count_total'] == 3


def test_cli_skips_blank_lines(tmp_path, capsys):
    data_file = tmp_path / 'episode_blank.jsonl'
    body = json.dumps(make_record()) + '\n\n' + json.dumps(make_record()) + '\n'
    data_file.write_text(body, encoding='utf-8')
    exit_code = main([str(data_file), '--expected-steps', '2'])
    captured = capsys.readouterr()
    assert exit_code == 0
    result = json.loads(captured.out)
    assert result['valid'] is True
    assert result['steps_completed'] == 2


def test_cli_rejects_invalid_records(tmp_path, capsys):
    data_file = tmp_path / 'bad_records.jsonl'
    data_file.write_text(json.dumps({'dss_converged': True}) + '\n', encoding='utf-8')
    exit_code = main([str(data_file), '--expected-steps', '1'])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ''
    assert '错误' in captured.err


def test_cli_rejects_non_object_line(tmp_path, capsys):
    data_file = tmp_path / 'bad_line.jsonl'
    data_file.write_text('[1, 2]\n', encoding='utf-8')
    exit_code = main([str(data_file), '--expected-steps', '1'])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert 'JSON 对象' in captured.err


def test_cli_missing_file(tmp_path, capsys):
    exit_code = main([str(tmp_path / 'nope.jsonl'), '--expected-steps', '1'])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert '错误' in captured.err

def test_native_divergence_record_without_dss_flag():
    result = summarize_episode([{'physical_failure': True}], 24)
    assert result['status'] == 'solver_failure'
    assert result['failure']['reason'] == 'physical_failure=true'
    assert result['energy_kwh'] is None


def test_conflicting_native_flags_raise():
    with pytest.raises(ValueError, match='physical_failure'):
        summarize_episode([{'physical_failure': True, 'dss_converged': True}], 24)


def test_voltage_duration_uses_unequal_intervals_not_phase_hours():
    records = [
        make_record(step_seconds=900, voltage_violation_count=4,
                    voltage_min_pu=.91, voltage_max_pu=1.01),
        make_record(step_seconds=2700, voltage_violation_count=0,
                    voltage_min_pu=.95, voltage_max_pu=1.05),
        make_record(step_seconds=1200, voltage_violation_count=1,
                    voltage_min_pu=.99, voltage_max_pu=1.10),
    ]
    result = summarize_episode(records, 3,
                               voltage_lower_limit_pu=.95,
                               voltage_upper_limit_pu=1.05)
    assert result["voltage_violation_count_total"] == 5
    assert result["voltage_violation_interval_hours"] == pytest.approx(2100 / 3600)
    assert result["voltage_max_exceedance_pu"] == pytest.approx(.05)
    assert not result["eligible_for_day_ranking"]
    assert result["day_ranking_refusal"] == "not_24_hours"


@pytest.mark.parametrize("failure,reason", [(False, "incomplete"), (True, "solver_failure")])
def test_voltage_diagnostics_not_rankable_on_incomplete_or_failed_day(failure, reason):
    records = [make_record(step_seconds=900, voltage_violation_count=3,
                           voltage_min_pu=.93, voltage_max_pu=1.01)]
    if failure:
        records.append({"physical_failure": True})
    result = summarize_episode(records, 24,
                               voltage_lower_limit_pu=.95,
                               voltage_upper_limit_pu=1.05)
    assert result["voltage_violation_interval_hours"] is None
    assert result["voltage_max_exceedance_pu"] is None
    assert result["diagnostic_voltage_violation_interval_hours"] == .25
    assert result["diagnostic_voltage_max_exceedance_pu"] == pytest.approx(.02)
    assert result["energy_kwh"] is None
    assert not result["eligible_for_day_ranking"]
    assert result["day_ranking_refusal"] == reason


def test_complete_day_and_no_default_limits():
    result = summarize_episode([make_record(step_seconds=3600)] * 24, 24)
    assert result["eligible_for_day_ranking"]
    assert result["day_ranking_refusal"] is None
    assert result["voltage_limits_pu"] is None
    assert result["voltage_max_exceedance_pu"] is None


@pytest.mark.parametrize("lower,upper", [(None, 1.05), (.95, None),
                                        (1.05, .95), (True, 1.05),
                                        (.95, float("inf"))])
def test_invalid_limits_refused(lower, upper):
    with pytest.raises(ValueError):
        summarize_episode([], 24, voltage_lower_limit_pu=lower,
                          voltage_upper_limit_pu=upper)


def test_native_count_and_explicit_limits_must_agree():
    with pytest.raises(ValueError, match="contradicts"):
        summarize_episode([make_record(voltage_violation_count=0,
                                       voltage_min_pu=.94, voltage_max_pu=1.02)],
                          1, voltage_lower_limit_pu=.95, voltage_upper_limit_pu=1.05)
