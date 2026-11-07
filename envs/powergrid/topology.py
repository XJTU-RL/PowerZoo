"""当前原生导通状态、有效电压源与相节点供电连通；不把DER名称当电源。"""

import networkx as nx


def terminal_nodes(info, terminal):
    bus = info["buses"][terminal].split(".")[0].lower()
    start = terminal * info["conductors"]
    return [
        (bus, node)
        for node in info["nodes"][start : start + info["conductors"]]
        if node > 0
    ]


def is_closed(info):
    return all(not opened for states in info["open"] for opened in states)


def branch_connections(session, switches=None):
    result = {}
    for element in sorted(session.native.AllElementNames):
        if not element.lower().startswith(("line.", "transformer.")):
            continue
        info = session.connection(element)
        if not info["enabled"]:
            continue
        if switches is not None and element in switches:
            if not switches[element]:
                continue
            info["open"] = [[False] * info["conductors"] for _ in info["buses"]]
        result[element] = info
    return result


def operating_graph(session, switches=None):
    """母线级径向操作约束：单相调压器组并为一条连接，重叠相并联支路仍成环。"""
    graph = nx.Graph()
    graph.add_nodes_from(session.bus_names)
    parallel = False
    for element, info in branch_connections(session, switches).items():
        terminals = []
        conductive = None
        if element.lower().startswith("line."):
            n = info["conductors"]
            conductive = {
                i
                for i in range(n)
                if info["nodes"][i] > 0
                and info["nodes"][n + i] > 0
                and not info["open"][0][i]
                and not info["open"][1][i]
            }
        for i in range(info["terminals"]):
            start = i * info["conductors"]
            available = {
                node
                for j, node in enumerate(
                    info["nodes"][start : start + info["conductors"]]
                )
                if node > 0
                and not info["open"][i][j]
                and (conductive is None or j in conductive)
            }
            if available:
                terminals.append((info["buses"][i].split(".")[0].lower(), available))
        for bus, phases in terminals[1:]:
            first, first_phases = terminals[0]
            shared = first_phases | phases
            if graph.has_edge(first, bus):
                edge = graph[first][bus]
                if edge["phases"] & shared:
                    parallel = True
                edge["phases"].update(shared)
                edge["elements"].append(element)
            else:
                graph.add_edge(first, bus, phases=set(shared), elements=[element])
    graph.graph["radial"] = not parallel and nx.is_forest(graph)
    return graph


def source_nodes(session):
    roots = set()
    for element in session.native.AllElementNames:
        if not element.lower().startswith("vsource."):
            continue
        info = session.connection(element)
        if info["enabled"] and float(session.read_property(element, "pu")) > 0:
            for i, node in enumerate(info["nodes"][: info["conductors"]]):
                if node > 0 and not info["open"][0][i]:
                    roots.add((info["buses"][0].split(".")[0].lower(), node))
    return roots


def energized_phases(session):
    """线路按实际导体传递；变压器要求一个完整有源绕组，暂不模拟缺相传递。"""
    reached = source_nodes(session)
    branches = branch_connections(session)
    changed = True
    while changed:
        old = len(reached)
        for element, info in branches.items():
            if element.lower().startswith("line."):
                n = info["conductors"]
                for i in range(n):
                    a, b = info["nodes"][i], info["nodes"][n + i]
                    if a <= 0 or b <= 0 or info["open"][0][i] or info["open"][1][i]:
                        continue
                    u = (info["buses"][0].split(".")[0].lower(), a)
                    v = (info["buses"][1].split(".")[0].lower(), b)
                    if u in reached or v in reached:
                        reached.update((u, v))
            else:
                windings = [
                    set(terminal_nodes(info, t))
                    for t in range(info["terminals"])
                    if not any(info["open"][t])
                ]
                if any(nodes and nodes <= reached for nodes in windings):
                    for nodes in windings:
                        reached.update(nodes)
        changed = len(reached) != old
    return reached
