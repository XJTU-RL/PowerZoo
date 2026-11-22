"""Conservative repeated-action capability witnesses in independent DSS states."""


def repeated_witnesses(restore, table_size, horizons=(1,2,5)):
    """Try a bounded family of constant setpoints, then independently execute winners."""
    paths = {}
    for idx in range(table_size):
        steps = []
        with restore() as plant:
            for _ in range(max(horizons)):
                row = plant.probe(idx)
                if row["veto"]:
                    break
                steps.append(plant.execute(idx)["delivered_kw"])
        paths[idx] = steps
    result = {}
    for n in horizons:
        feasible = [(min(values[:n]),idx) for idx,values in paths.items() if len(values)>=n]
        if not feasible:
            result[n] = dict(confirmed=False, capacity_kw=None)
            continue
        cap,idx = max(feasible)
        with restore() as plant:
            actual = [plant.execute(idx)["delivered_kw"] for _ in range(n)]
        if abs(min(actual)-cap)>1e-5:
            raise ValueError("independent sustained witness disagrees")
        result[n] = dict(confirmed=True,capacity_kw=min(cap,min(actual)), action=idx, independent_kw=actual)
    return result,paths
