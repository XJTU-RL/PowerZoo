"""实际有功注入/吸收，所有条形量均为kW；不混入缩放无功。"""

import plotly.graph_objects as go
from envs.render_common.viz.theme import get_plotly_layout


def create_power_flow_diagram(snapshot):
    c = snapshot["circuit"]
    rows = [
        ("Grid", c["grid_import_kw"]),
        ("Load", -c["total_load_kw"]),
        ("Losses", -c["total_loss_kw"]),
    ]
    for kind, prefix in (("pvs", "PV"), ("batteries", "Storage")):
        rows.extend(
            (prefix + " " + name, device["power_kw"])
            for name, device in snapshot["devices"][kind].items()
        )
    fig = go.Figure(
        go.Bar(x=[v for _, v in rows], y=[name for name, _ in rows], orientation="h")
    )
    fig.update_layout(
        **get_plotly_layout(
            title="Active Power: Injection Positive", height=400, env_name="vvc"
        )
    )
    fig.update_xaxes(title_text="kW")
    return fig
