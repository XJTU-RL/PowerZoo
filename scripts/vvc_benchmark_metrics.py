#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''VVC（Volt/VAR Control，伏瓦控制）单回合评测指标汇总助手。

窄口径离线工具：汇总 XJTU-RL/PowerZoo（基线 commit de8ce3c7）native 评测
中「一个回合」的保守指标。

范围与红线：
- 只消费 envs/powergrid/scenario.py 已冻结且互不相交的既有
  train/validation/test 清单所对应的回合记录；不新建清单、不改环境。
- 所有指标仅刻画单个回合；仅服务于 validation 切分上的 checkpoint 选取，
  不产生测试集结论或跨回合/跨数据集声明。
- MAPPO 与 SHOM 均未验证；本工具不添加策略、训练、网络调用或 native 控制。
- 计数保持计数，不折算速率；不从 step_seconds 推断回合时长。
- 仅依赖 Python 3.10 标准库，确定性计算，仅本地文件读写，无网络。

命令行用法：

    python scripts/vvc_benchmark_metrics.py episode.jsonl --expected-steps 240
'''

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ['build_parser', 'main', 'summarize_episode']

_MISSING = object()


def _require_bool(record: Mapping[str, Any], index: int, key: str) -> bool:
    value = record.get(key, _MISSING)
    if value is _MISSING:
        raise ValueError(f'记录 {index}：缺少字段 {key}')
    if not isinstance(value, bool):
        raise ValueError(f'记录 {index}：字段 {key} 必须为布尔值，实际类型为 {type(value).__name__}')
    return value


def _require_number(
    record: Mapping[str, Any],
    index: int,
    key: str,
    *,
    integer: bool = False,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    value = record.get(key, _MISSING)
    if value is _MISSING:
        raise ValueError(f'记录 {index}：缺少字段 {key}')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'记录 {index}：字段 {key} 必须为数值，实际类型为 {type(value).__name__}')
    if integer and not isinstance(value, int):
        raise ValueError(f'记录 {index}：字段 {key} 必须为整数，实际为 {value!r}')
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f'记录 {index}：字段 {key} 数值超出浮点可表示范围') from None
    if not math.isfinite(number):
        raise ValueError(f'记录 {index}：字段 {key} 必须为有限数值，实际为 {value!r}')
    if minimum is not None:
        violates = number <= minimum if exclusive_minimum else number < minimum
        if violates:
            bound = '大于' if exclusive_minimum else '不小于'
            raise ValueError(f'记录 {index}：字段 {key} 必须{bound} {minimum}，实际为 {value!r}')
    return value if integer else number


def summarize_episode(records: Iterable[Mapping[str, Any]], expected_steps: int) -> dict[str, Any]:
    '''汇总单个 VVC 评测回合，返回 JSON 可序列化的指标字典。

    参数：
        records：按时间顺序排列的记录映射（来自 native 评测信息）。每条必须
            含布尔字段 dss_converged，或原生失败标记 physical_failure=True；收敛步还必须提供 step_seconds（本步
            实际步长区间秒数，正有限，允许逐步不同）、power_loss_kw（非负
            有限）、voltage_violation_count（非负整数）、voltage_min_pu 与
            voltage_max_pu（有限且 min 不大于 max）。其余字段忽略。首个
            dss_converged 为 False 的步视为求解失败：该步其余字段允许缺省
            （不会被读取），处理在此停下并打标记；其后的记录一律不再消费，
            仅在 failure.trailing_records_ignored 中报告被忽略的条数。
        expected_steps：完整回合应有的步数（正整数）。

    返回字段：
        valid / status / expected_steps / steps_completed / steps_processed /
        energy_kwh / diagnostic_energy_kwh / voltage_min_pu / voltage_max_pu /
        voltage_violation_count_total / step_seconds_min / step_seconds_mean /
        step_seconds_max / failure

    语义：
        - status 取 valid（记录数与 expected_steps 相等且全部收敛）、
          incomplete（记录不足且无失败步）或 solver_failure（出现失败步）。
        - energy_kwh 仅在 valid 时给出，否则为 None；diagnostic_energy_kwh
          始终为收敛前缀的部分能量（诊断用）。能量按每步实际区间积分：
          kWh = Σ power_loss_kw × step_seconds / 3600。
        - 失败步不计入能量、电压、越限计数与 step_seconds 统计。
        - voltage_violation_count_total 是原始计数的求和，不是速率；
          不输出回合时长，step_seconds_min/mean/max 仅为每步观测值。

    异常：
        ValueError：输入不合法（记录条数超过 expected_steps、布尔/数值/
        整数校验失败、电压上下限次序错误、记录不是映射等）。
    '''
    if isinstance(expected_steps, bool) or not isinstance(expected_steps, int) or expected_steps <= 0:
        raise ValueError(f'expected_steps 必须为正整数，实际为 {expected_steps!r}')
    try:
        record_list = list(records)
    except TypeError as exc:
        raise ValueError('records 必须为记录映射的可迭代序列') from exc
    if len(record_list) > expected_steps:
        raise ValueError(f'记录条数 {len(record_list)} 超过 expected_steps={expected_steps}，拒绝处理超长输入')

    steps_completed = 0
    diagnostic_energy_kwh = 0.0
    voltage_min_pu = None
    voltage_max_pu = None
    voltage_violation_count_total = 0
    step_seconds_min = None
    step_seconds_max = None
    step_seconds_sum = 0.0
    failure = None

    for index, record in enumerate(record_list):
        if not isinstance(record, Mapping):
            raise ValueError(f'记录 {index}：必须为映射（dict），实际类型为 {type(record).__name__}')
        native_failure = (
            _require_bool(record, index, 'physical_failure')
            if 'physical_failure' in record else False
        )
        if native_failure and 'dss_converged' in record:
            if _require_bool(record, index, 'dss_converged'):
                raise ValueError(f'记录 {index}：physical_failure 与 dss_converged 矛盾')
        if native_failure or not _require_bool(record, index, 'dss_converged'):
            failure = {
                'step_index': index,
                'reason': 'physical_failure=true' if native_failure else 'dss_converged=false',
                'trailing_records_ignored': len(record_list) - index - 1,
            }
            break
        step_seconds = _require_number(record, index, 'step_seconds', minimum=0.0, exclusive_minimum=True)
        power_loss_kw = _require_number(record, index, 'power_loss_kw', minimum=0.0)
        violation_count = _require_number(record, index, 'voltage_violation_count', integer=True, minimum=0)
        v_min = _require_number(record, index, 'voltage_min_pu')
        v_max = _require_number(record, index, 'voltage_max_pu')
        if v_min > v_max:
            raise ValueError(f'记录 {index}：voltage_min_pu ({v_min}) 不应大于 voltage_max_pu ({v_max})')
        diagnostic_energy_kwh += power_loss_kw * (step_seconds / 3600.0)
        if not math.isfinite(diagnostic_energy_kwh):
            raise ValueError(f'记录 {index}：累计能量超出有限范围')
        voltage_violation_count_total += violation_count
        voltage_min_pu = v_min if voltage_min_pu is None else min(voltage_min_pu, v_min)
        voltage_max_pu = v_max if voltage_max_pu is None else max(voltage_max_pu, v_max)
        step_seconds_min = step_seconds if step_seconds_min is None else min(step_seconds_min, step_seconds)
        step_seconds_max = step_seconds if step_seconds_max is None else max(step_seconds_max, step_seconds)
        step_seconds_sum += step_seconds
        if not math.isfinite(step_seconds_sum):
            raise ValueError(f'记录 {index}：累计步长超出有限范围')
        steps_completed += 1

    if failure is not None:
        status = 'solver_failure'
    elif steps_completed < expected_steps:
        status = 'incomplete'
    else:
        status = 'valid'

    return {
        'valid': status == 'valid',
        'status': status,
        'expected_steps': expected_steps,
        'steps_completed': steps_completed,
        'steps_processed': steps_completed + (0 if failure is None else 1),
        'energy_kwh': diagnostic_energy_kwh if status == 'valid' else None,
        'diagnostic_energy_kwh': diagnostic_energy_kwh,
        'voltage_min_pu': voltage_min_pu,
        'voltage_max_pu': voltage_max_pu,
        'voltage_violation_count_total': voltage_violation_count_total,
        'step_seconds_min': step_seconds_min,
        'step_seconds_mean': step_seconds_sum / steps_completed if steps_completed else None,
        'step_seconds_max': step_seconds_max,
        'failure': failure,
    }


def _load_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path} 第 {line_number} 行不是合法 JSON：{exc}') from exc
            if not isinstance(payload, dict):
                raise ValueError(f'{path} 第 {line_number} 行必须是 JSON 对象')
            records.append(payload)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='vvc_benchmark_metrics',
        description='离线汇总单个 VVC 评测回合（JSONL）的指标；仅本地读写，无网络。',
    )
    parser.add_argument('jsonl_path', help='回合记录的 JSONL 文件路径（每行一个 JSON 对象，空行跳过）')
    parser.add_argument('--expected-steps', type=int, required=True, help='完整回合应有的步数（正整数）')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = _load_records(args.jsonl_path)
        summary = summarize_episode(records, args.expected_steps)
    except (OSError, ValueError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
