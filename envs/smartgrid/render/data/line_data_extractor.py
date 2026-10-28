"""支路数据来自同一次物理状态；长度按原生单位转换，未知单位保持None。"""

from envs.powergrid.snapshot import (
    extract_line_data,
    extract_transformer_data,
    extract_edge_list,
)

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
