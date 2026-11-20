import pytest

from scripts.check_external_temporal_parity import (
    OEDI_SCRIPTS,
    OEDI_UNSHAPED,
    aligned_endpoint,
    day_indices,
    ensure_finite,
    max_tap_error,
    max_phase_error,
    oedi_input_digest,
    parse_sensor_nodes,
    sensor_projection,
    shape_definitions,
    verify_unshaped,
)


def deck(root, declarations):
    qsts = root / "qsts"
    qsts.mkdir()
    for filename, declaration in zip(OEDI_SCRIPTS, declarations):
        (qsts / filename).write_text(declaration)


def test_day_index_and_yearly_native_interval_alignment():
    assert list(day_indices(0)) == list(range(96))
    assert day_indices(364)[-1] == 35039
    for index in (0, 1, 95, 96, 35039):
        aligned_endpoint(index, (index + 1) * 900.0)
        with pytest.raises(ValueError, match="clock"):
            aligned_endpoint(index, index * 900.0)
    for day in (-1, 365, True):
        with pytest.raises(ValueError, match="zero-based"):
            day_indices(day)


def test_declared_p_q_and_temperature_paths_stay_separate(tmp_path):
    deck(
        tmp_path,
        (
            "New Tshape.t npts=35040 interval=0.25 temp=(file=../profiles/T.csv)",
            "New Loadshape.p npts=35040 interval=0.25 mult=(file=../profiles/PV.csv)",
            "New Loadshape.l npts=35040 interval=0.25 mult=(file=../profiles/P.csv) qmult=(file=../profiles/Q.csv)",
        ),
    )
    rows = shape_definitions(tmp_path)
    assert rows["l"] == {
        "mult": tmp_path / "profiles/P.csv", "qmult": tmp_path / "profiles/Q.csv"
    }
    assert rows["t"] == {"temp": tmp_path / "profiles/T.csv"}
    assert rows["p"] == {"mult": tmp_path / "profiles/PV.csv"}


@pytest.mark.parametrize(
    "properties, expected",
    [
        ("npts=96 interval=0.25 mult=(file=../profiles/PV.csv)", "interval or length"),
        ("npts=35040 interval=0.5 mult=(file=../profiles/PV.csv)", "interval or length"),
        ("npts=35040 interval=0.25 mult=(file=../../PV.csv)", "escapes"),
    ],
)
def test_rejects_unsupported_shape_records(tmp_path, properties, expected):
    deck(tmp_path, ("", "New Loadshape.p " + properties, ""))
    with pytest.raises(ValueError, match=expected):
        shape_definitions(tmp_path)


def test_unshaped_only_verified_inventory_and_nominal_power():
    verify_unshaped(OEDI_UNSHAPED)
    with pytest.raises(ValueError, match="unshaped"):
        verify_unshaped({**OEDI_UNSHAPED, "mystery": (1, 2)})
    with pytest.raises(ValueError, match="unshaped"):
        verify_unshaped({**OEDI_UNSHAPED, "s48": (0, 0)})


def test_compares_node_identity_independent_of_iteration_order():
    assert max_phase_error({"a.1": 1, "b.2": 1.1}, {"b.2": 1, "a.1": 1}) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="phase nodes"):
        max_phase_error({"a.1": 1}, {"a.2": 1})



def test_oedi_hash_fails_for_changed_bytes_and_symlinks(tmp_path):
    a = tmp_path / "source"
    b = tmp_path / "relocated"
    a.mkdir()
    b.mkdir()
    (a / "profile.csv").write_bytes(b"1\r\n")
    (b / "profile.csv").write_bytes(b"1\r\n")
    assert oedi_input_digest(a) == oedi_input_digest(b)
    (b / "profile.csv").write_text("1\n")
    assert oedi_input_digest(a) != oedi_input_digest(b)
    (b / "linked").symlink_to(a / "profile.csv")
    with pytest.raises(ValueError, match="symlinks"):
        oedi_input_digest(b)



def test_reject_nonfinite_measurements_and_clock():
    with pytest.raises(ValueError, match="nonfinite"):
        ensure_finite([1, float("nan")])
    with pytest.raises(ValueError, match="clock"):
        aligned_endpoint(0, float("nan"))


def test_static_tap_comparison_fails_if_inventory_differs():
    assert max_tap_error({"r1": 1, "r2": -2}, {"r2": -2, "r1": 1}) == 0
    assert max_tap_error({"r1": 1}, {"r1": 4}) == 3
    with pytest.raises(ValueError, match="inventory"):
        max_tap_error({"r1": 1}, {})



def test_sensor_projection_respects_bus_phase_order_and_exact_identity():
    labels = parse_sensor_nodes(b'["BusA.2","busa.1","busB.3"]')
    assert list(sensor_projection({"busa.1": 1, "busa.2": 0.99, "busb.3": 1.01}, labels)) == [
        "busa.2", "busa.1", "busb.3"
    ]
    with pytest.raises(ValueError, match="absent"):
        sensor_projection({"busa.1": 1}, ("busa.2",))


@pytest.mark.parametrize("payload", [b'["7.0"]', b'["7.1","7.1"]', b'["7"]', b'{}'])
def test_rejects_invalid_sensor_list(payload):
    with pytest.raises(ValueError):
        parse_sensor_nodes(payload)
