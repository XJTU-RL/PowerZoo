"""Runner registry."""
from runners.on_policy_ha_runner import OnPolicyHARunner
from runners.on_policy_ma_runner import OnPolicyMARunner
from runners.off_policy_ha_runner import OffPolicyHARunner
from runners.off_policy_ma_runner import OffPolicyMARunner
from runners.Qmix_runner import QMIXRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "qmix": QMIXRunner,
    "shom": OnPolicyHARunner,
}
