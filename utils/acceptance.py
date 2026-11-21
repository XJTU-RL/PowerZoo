"""训练器无关的分阶段验收。运行器通过回调提交本次验收的证据。"""

from dataclasses import dataclass
import math
from pathlib import Path
import json
import re
from typing import Callable

from utils.capabilities import Preflight

STAGES = (
    "preflight",
    "physical_readback",
    "minimal_update",
    "save_reload",
    "independent_replay",
)


class AcceptanceBlocked(RuntimeError):
    """缺少运行依赖、配置或输入，验收暂停并保留此前证据。"""


@dataclass(frozen=True)
class AcceptanceChecks:
    physical_readback: Callable[[], dict]
    minimal_update: Callable[[], dict]
    save_reload: Callable[[], dict]
    independent_replay: Callable[[], dict]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive_int(value):
    return type(value) is int and value > 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _readback_matches(row):
    values = [row[key] for key in ("requested", "applied", "measured", "tolerance")]
    return (
        isinstance(row["quantity"], str)
        and bool(row["quantity"])
        and all(_finite(value) for value in values)
        and row["tolerance"] >= 0
        and abs(row["applied"] - row["measured"]) <= row["tolerance"]
        and row["matched"] is True
    )


def _validate_evidence(stage, evidence, prior):
    _require(isinstance(evidence, dict), "阶段证据须为 JSON 对象")
    if stage == "physical_readback":
        _require(evidence["converged"] is True, "原生潮流未收敛")
        error, tolerance = (
            evidence["active_balance_error_kw"],
            evidence["balance_tolerance_kw"],
        )
        _require(
            _finite(error)
            and _finite(tolerance)
            and 0 < tolerance
            and abs(error) <= tolerance,
            "有功不平衡或容差无效",
        )
        ids = evidence["device_ids"]
        readings = evidence["readback"]
        _require(
            isinstance(ids, (list, tuple))
            and bool(ids)
            and all(isinstance(name, str) and name for name in ids)
            and len({name.casefold() for name in ids}) == len(ids),
            "需要唯一的已绑定设备 ID",
        )
        _require(
            isinstance(readings, dict)
            and set(ids) == set(readings)
            and all(_readback_matches(row) for row in readings.values()),
            "每个绑定设备都须有匹配的原生命令读回",
        )
    elif stage == "minimal_update":
        _require(
            _positive_int(evidence["steps"]) and _positive_int(evidence["updates"]),
            "需要实际环境步和至少一次更新",
        )
        before, after = (
            evidence["parameters_before_sha256"],
            evidence["parameters_after_sha256"],
        )
        _require(
            _digest(before) and _digest(after) and before != after, "策略参数未更新"
        )
        scenarios = evidence["training_scenarios"]
        _require(
            isinstance(scenarios, dict)
            and bool(scenarios)
            and all(
                isinstance(key, str) and key and _digest(value)
                for key, value in scenarios.items()
            ),
            "需要本次训练实际消费的场景 ID 及摘要",
        )
    elif stage == "save_reload":
        _require(_digest(evidence["checkpoint_sha256"]), "需要检查点摘要")
        _require(evidence["prediction_match"] is True, "重载前后确定性预测不一致")
    elif stage == "independent_replay":
        training = prior["minimal_update"]["training_scenarios"]
        _require(
            evidence["split"] == "validation"
            and isinstance(evidence["scenario_id"], str)
            and bool(evidence["scenario_id"])
            and evidence["scenario_id"] not in training
            and _digest(evidence["scenario_sha256"])
            and evidence["scenario_sha256"] not in training.values(),
            "验证工况必须来自 validation，且 ID 和内容均不得复用训练工况",
        )
        _require(
            evidence["checkpoint_sha256"] == prior["save_reload"]["checkpoint_sha256"],
            "回放须使用本次重载的检查点",
        )
        _require(
            _positive_int(evidence["steps"])
            and _positive_int(evidence["expected_steps"])
            and evidence["steps"] == evidence["expected_steps"]
            and evidence["terminated"] is True
            and evidence["truncated"] is False
            and evidence["physical_failure"] is False,
            "独立工况须完成全部步骤并自然终止",
        )


def run_acceptance(preflight: Preflight, checks: AcceptanceChecks | None = None):
    """为一个已解析组合执行验收，首个失败/阻塞会跳过后续阶段。"""
    report = {"schema_version": 1, "stages": {}}
    stages = report["stages"]
    stages["preflight"] = {
        "status": "passed" if preflight.compatible else "failed",
        "evidence": preflight.to_dict(),
    }
    evidence_by_stage = {}
    proceed = preflight.compatible
    for stage in STAGES[1:]:
        if not proceed or checks is None:
            stages[stage] = {"status": "not_run"}
            continue
        evidence = None
        try:
            result = getattr(checks, stage)()
            json.dumps(result, allow_nan=False)
            if isinstance(result, dict):
                evidence = result
            _validate_evidence(stage, evidence, evidence_by_stage)
            evidence_by_stage[stage] = evidence
            stages[stage] = {"status": "passed", "evidence": evidence}
        except AcceptanceBlocked as exc:
            stages[stage] = {"status": "blocked", "detail": str(exc)}
            proceed = False
        except Exception as exc:
            stages[stage] = {
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
            }
            if evidence is not None:
                stages[stage]["evidence"] = evidence
            proceed = False

    statuses = {entry["status"] for entry in stages.values()}
    report["status"] = next(
        status
        for status in ("failed", "blocked", "not_run", "passed")
        if status in statuses
    )
    return report


def write_report(report, path):
    """写入调用方新建的报告路径。"""
    report = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(report + "\n")
