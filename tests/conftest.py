import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    from medminer.core.sample_data import write_samples
    d = tmp_path_factory.mktemp("samples")
    write_samples(d)
    return d


@pytest.fixture(scope="session")
def heart_df():
    from medminer.core.sample_data import make_heart_sample
    return make_heart_sample()


@pytest.fixture
def heart_ds(heart_df):
    from medminer.core.dataset import Dataset
    return Dataset.from_frame("heart", heart_df.copy())
