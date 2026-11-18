import os
import subprocess
import sys

import numpy as np
import pytest


DECK = """
[problem]
name = "turbulence"

[mesh]
nx = 16
ny = 16

[time]
tlim = 0.3

[output]
file = "{name}.npz"
snapshot_dt = {snapshot_dt}
"""


def run(tmp_path, name, snapshot_dt):
    deck = tmp_path / f"{name}.toml"
    deck.write_text(DECK.format(name=name, snapshot_dt=snapshot_dt))
    env = dict(os.environ, JAX_ENABLE_X64="1")
    env.setdefault("JAX_PLATFORMS", "cpu")
    subprocess.run([sys.executable, "-m", "jathenax.run", "-i", str(deck)],
                   cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    with np.load(tmp_path / f"{name}.npz") as data:
        return data["conserved"], data["time"], data["cycle"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("snapshots")
    # 0.1 is not exact in binary, so 3 * 0.1 > 0.3 = tlim: the last snapshot needs the tolerance.
    return tmp_path, run(tmp_path, "plain", 0.0), run(tmp_path, "snaps", 0.1)


def test_snapshots_leave_the_evolution_unchanged(runs):
    _, plain, snaps = runs
    for a, b in zip(plain, snaps):
        np.testing.assert_array_equal(a, b)


def test_one_snapshot_per_scheduled_time(runs):
    tmp_path, _, (final, _, final_cycle) = runs
    assert not (tmp_path / "plain").exists()
    paths = sorted((tmp_path / "snaps").glob("*.npz"))
    assert [p.name for p in paths] == [f"snap.{n:05d}.npz" for n in range(4)]
    previous = -1
    for n, path in enumerate(paths):
        with np.load(path) as data:
            # Written after the first step that reaches n * 0.1, so at most one step late.
            assert data["time"] >= n * 0.1 - 1e-12 and data["cycle"] > previous
            assert data["dens"].shape == (16, 16) and data["dens"].dtype == np.float64
            previous = data["cycle"]
    # The last snapshot is at tlim and is the final state.
    with np.load(paths[-1]) as data:
        assert data["time"] == 0.3 and data["cycle"] == final_cycle
        np.testing.assert_array_equal(data["dens"], final[..., 0])
        np.testing.assert_array_equal(data["velx"], final[..., 1] / final[..., 0])
        np.testing.assert_array_equal(data["vely"], final[..., 2] / final[..., 0])
