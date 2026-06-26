from __future__ import annotations

import numpy as np

from sptoolkit.deconv.spectrum_io import find_csv_files, read_spectrum


def test_find_csv_files_creates_missing_input_dir(tmp_path):
    input_dir = tmp_path / "in"

    files = find_csv_files(input_dir)

    assert files == []
    assert input_dir.is_dir()


def test_read_spectrum_sorts_and_drops_invalid_rows(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text(
        "shift,intensity\n"
        "3,30\n"
        "bad,40\n"
        "1,10\n"
        "2,\n",
        encoding="utf-8",
    )

    x, y = read_spectrum(path)

    np.testing.assert_allclose(x, [1.0, 3.0])
    np.testing.assert_allclose(y, [10.0, 30.0])
