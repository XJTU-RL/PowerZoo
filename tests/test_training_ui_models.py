"""Model discovery follows the layouts produced by both training runners."""

from training_ui.core.model_discovery import filter_models, scan_models


def test_scan_models_finds_legacy_and_current_runner_outputs(tmp_path):
    files = {
        "vvc/13Bus/happo/trial/seed-001/models/actor_agent0.pt": "",
        "vvc/13Bus/happo/trial/seed-001/models/checkpoint_episode_1/actor_agent0.pt": "",
        "vvc/13Bus/happo/trial/seed-001/config.json": '{"legacy": true}',
        "vvc/13Bus/qmix/trial/seed-002/models/mixer.pt": "",
        "vvc/13Bus/qmix/trial/seed-002/models/policy_0/q_network.pt": "",
        "stackelberg/pricing/run_42/checkpoint/config.resolved.yaml": (
            "env: {system_name: PricingSmall}"
        ),
        "stackelberg/pricing/run_42/checkpoint/agent_0.zip": "",
        "unrelated/a/b/c/d/notes.txt": "",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    records = scan_models(str(tmp_path))
    assert len(records) == 3

    [vvc] = filter_models(records, env="vvc", algo="happo")
    assert vvc.config == {"legacy": True}
    assert vvc.checkpoints == [
        "models/actor_agent0.pt",
        "models/checkpoint_episode_1/actor_agent0.pt",
    ]

    [qmix] = filter_models(records, algo="qmix")
    assert qmix.checkpoints == ["models/mixer.pt", "models/policy_0/q_network.pt"]

    [pricing] = filter_models(records, algo="sequential_ppo")
    assert pricing.system == "PricingSmall"
    assert pricing.config == {"env": {"system_name": "PricingSmall"}}
    assert pricing.checkpoints == ["checkpoint/agent_0.zip"]
