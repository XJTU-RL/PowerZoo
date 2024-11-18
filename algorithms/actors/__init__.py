"""Algorithm registry."""
from algorithms.actors.happo import HAPPO
from algorithms.actors.hatrpo import HATRPO
from algorithms.actors.haa2c import HAA2C
from algorithms.actors.haddpg import HADDPG
from algorithms.actors.hatd3 import HATD3
from algorithms.actors.hasac import HASAC
from algorithms.actors.had3qn import HAD3QN
from algorithms.actors.maddpg import MADDPG
from algorithms.actors.matd3 import MATD3
from algorithms.actors.mappo import MAPPO
from algorithms.actors.shom import SHOM

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
    "shom": SHOM,
}
