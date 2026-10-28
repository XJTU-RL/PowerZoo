"""支路数据来自同一次物理状态；长度按原生单位转换，未知单位保持None。"""

_LENGTH_TO_KM = {
    1: 1.609344,
    2: 0.3048,
    3: 1.0,
    4: 0.001,
    5: 0.0003048,
    6: 0.0000254,
    7: 0.00001,
    8: 0.000001,
}


def extract_line_data(env):
    result = {}
    for element, info in env.physical_state.branches.items():
        if not element.lower().startswith("line."):
            continue
        name = element.split(".", 1)[1]
        lines = env.circuit.native.Lines
        lines.Name = name
        length, units = float(lines.Length), int(lines.Units)
        result[name] = {
            "name": name,
            "type": "line",
            "bus1": info["buses"][0].split(".")[0],
            "bus2": info["buses"][1].split(".")[0],
            "length_native": length,
            "length_units": units,
            "length_km": (
                length * _LENGTH_TO_KM[units] if units in _LENGTH_TO_KM else None
            ),
            "phases": info["phases"],
            "loss_kw": info["losses"][0],
            "loss_kvar": info["losses"][1],
            "loading_pct": (
                info["loading_pu"] * 100 if info["loading_pu"] is not None else None
            ),
        }
    return result


def extract_transformer_data(env):
    return {
        element.split(".", 1)[1]: {
            "name": element.split(".", 1)[1],
            "type": "transformer",
            "bus1": info["buses"][0].split(".")[0],
            "bus2": info["buses"][1].split(".")[0],
            "kva": info["winding_kva"][0],
            "loss_kw": info["losses"][0],
            "phases": info["phases"],
            "loading_pct": info["loading_pu"] * 100,
        }
        for element, info in env.physical_state.branches.items()
        if element.lower().startswith("transformer.")
    }


def extract_edge_list(env):
    return [
        (v["bus1"], v["bus2"], v["type"])
        for v in {**extract_line_data(env), **extract_transformer_data(env)}.values()
    ]
