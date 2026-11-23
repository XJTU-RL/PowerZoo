from .replay_capacity_transmission import injections, measure


def test_external_call_survives_capacity_shortfall():
    episode = dict(n_sites=3, steps=[dict(request_kw=10., activation=.5,
                    target_kw=[1., 1., 1.], delivered_kw=[1., 2., 0.], q_support_kvar=[.1, .2, -.3])])
    specs = {s: dict(anchor_kw_kvar=[20., 10.]) for s in "ABC"}
    rows = injections(episode, specs)
    assert rows == [dict(A=[24., 9.9], B=[18., 9.8], C=[20., 10.3])]
    assert all(spec["anchor_kw_kvar"] == [20., 10.] for spec in specs.values())


def test_frequency_iae_uses_same_time_neutral():
    ref = dict(writes_s=[[20., 21.]], bus8_time_hz=[[20., 59.9], [21., 59.8]])
    actual = dict(writes_s=[[20., 21.]], bus8_time_hz=[[20., 59.9], [21., 59.7]])
    assert abs(measure(actual, ref)["iae_from_neutral_hz_s"] - .05) < 1.e-10
