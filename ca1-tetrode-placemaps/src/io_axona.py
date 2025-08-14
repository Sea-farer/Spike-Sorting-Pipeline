"""

This module provides:
  - find_session_files(session_dir)
  - load_pos_axona(pos_path): parse Axona .pos (binary) into a DataFrame
  - load_units_t64(session_dir): parse .t64/TT*.t64 into spike times (sec)
  - load_units_mclust(session_dir): optional wrapper for scripts/mclust.py if present
"""

from pathlib import Path
import numpy as np
import pandas as pd
import re
from typing import Dict, Tuple

# Optional deps for .t64 via MATLAB format
try:
    from scipy.io import loadmat as _scipy_loadmat  # type: ignore
except Exception:  # pragma: no cover
    _scipy_loadmat = None
try:
    import h5py as _h5py  # type: ignore
except Exception:  # pragma: no cover
    _h5py = None

ARENA_CM = 150
SPEED_MIN, SPEED_MAX = 6, 60  # cm/s
SPK_FS = 48000.0  # Hz, Axona timestamp base for some formats (not used with MClust .t/.t64)


def find_session_files(session_dir: Path):
    session_dir = Path(session_dir)
    return {
        "pos": next(session_dir.glob("*.pos"), None),
        "set": next(session_dir.glob("*.set"), None),
        "t64_list": list(session_dir.glob("*_*.t64")) or list(session_dir.glob("TT*.t64")),
        "tetrodes": sum((list(session_dir.glob(f"*.{i}")) for i in range(1, 9)), []),
    }


def _split_pos_header_data(raw: bytes) -> Tuple[Dict[str, str], memoryview]:
    """Split raw .pos file bytes into (header_dict, data_bytes_view).

    Handles cases where 'data_start' is immediately followed by binary with no newline.
    """
    marker = b"data_start"
    idx = raw.find(marker)
    if idx == -1:
        raise RuntimeError("data_start not found in .pos header")
    header_text = raw[:idx].decode("latin1", errors="ignore")
    header: Dict[str, str] = {}
    for line in header_text.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split(" ", 1)
        if len(parts) == 1:
            header[parts[0]] = ""
        else:
            header[parts[0]] = parts[1]
    data_view = memoryview(raw)[idx + len(marker) :]
    return header, data_view


def _read_pos_binary(path: Path):
    """Return (t, x, y) from Axona .pos binary file."""
    raw = Path(path).read_bytes()
    header, data_view = _split_pos_header_data(raw)

    # Defaults with safe fallbacks
    sample_rate = float(header.get("sample_rate", "50").split()[0])
    num_colours = int(header.get("num_colours", "2"))
    b_ts = int(header.get("bytes_per_timestamp", "4"))
    b_coord = int(header.get("bytes_per_coord", "2"))
    pos_format = header.get("pos_format", "t,x1,y1,x2,y2")
    win_min_x = int(header.get("window_min_x", "0"))
    win_min_y = int(header.get("window_min_y", "0"))
    # pixel-to-centimeter conversion (pixels per metre -> cm per pixel)
    try:
        pixels_per_m = float(header.get("pixels_per_metre", "300").split()[0])
    except Exception:
        pixels_per_m = 300.0
    px_to_cm = 100.0 / pixels_per_m

    two_spot = ("x2" in pos_format and "y2" in pos_format)
    has_pix = ("numpix1" in pos_format) or ("numpix2" in pos_format)

    ts_dt = ">i" + str(b_ts)
    coord_dt = ">i" + str(b_coord)
    pix_dt = ">i4"

    if two_spot:
        fields = [("t", ts_dt), ("coords", coord_dt, 4)]
        if has_pix:
            fields.append(("pixel_count", pix_dt, 2))
    else:
        # Generalize: read up to 2 colour spots if present
        n_pairs = min(num_colours, 2)
        fields = [("t", ts_dt), ("coords", coord_dt, 2 * n_pairs)]
        if has_pix:
            fields.append(("pixel_count", pix_dt, n_pairs))

    dtype = np.dtype(fields)

    # Determine the exact binary slice to decode: stop at 'data_end' if present,
    # and truncate to full records (and to num_pos_samples if provided).
    raw = Path(path).read_bytes()
    start = raw.find(b"data_start") + len(b"data_start")
    end_marker = raw.find(b"data_end", start)
    if end_marker != -1:
        data_bytes = raw[start:end_marker]
    else:
        data_bytes = raw[start:]

    rec_size = dtype.itemsize
    n_full = len(data_bytes) // rec_size
    # Respect header num_pos_samples if it fits
    try:
        n_hdr = int(header.get("num_pos_samples", "0").strip().split()[0])
    except Exception:
        n_hdr = 0
    if n_hdr > 0 and n_hdr <= n_full:
        n_use = n_hdr
    else:
        n_use = n_full
    data = np.frombuffer(data_bytes[: n_use * rec_size], dtype=dtype)

    # Robust time from sample rate
    t = np.arange(len(data), dtype=float) / (sample_rate if sample_rate > 0 else 50.0)

    coords = data["coords"].astype(float)
    # Replace sentinel with NaN
    coords[coords == 1023] = np.nan

    # Choose best spot per sample
    if coords.shape[1] >= 4:
        x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    elif coords.shape[1] >= 2:
        x1, y1 = coords[:, 0], coords[:, 1]
        x2, y2 = np.full_like(x1, np.nan), np.full_like(y1, np.nan)
    else:
        raise RuntimeError(".pos file missing coordinate columns")

    use1 = ~np.isnan(x1) & ~np.isnan(y1)
    use2 = ~np.isnan(x2) & ~np.isnan(y2)
    x_px = np.where(use1, x1, np.where(use2, x2, np.nan)) + win_min_x
    y_px = np.where(use1, y1, np.where(use2, y2, np.nan)) + win_min_y
    x_cm = x_px * px_to_cm
    y_cm = y_px * px_to_cm
    return t, x_cm, y_cm


def load_pos_axona(pos_path: Path):
    """Parse Axona .pos into a DataFrame with columns t,x,y,ang,valid,speed,keep."""
    pos_path = Path(pos_path)
    t, x, y = _read_pos_binary(pos_path)

    # Valid: inside positive arena window with finite coords
    valid = np.isfinite(x) & np.isfinite(y) & (x >= -1) & (y >= -1) & (x <= ARENA_CM * 2) & (y <= ARENA_CM * 2)

    # Speed & keep mask
    dt = np.gradient(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0 / 50.0
    speed = np.hypot(np.gradient(x), np.gradient(y)) / dt

    df = pd.DataFrame({
        "t": t,
        "x": x,
        "y": y,
        "ang": np.nan,
        "valid": valid.astype(int),
        "speed": np.clip(speed, 0, None),
    })
    df["keep"] = (df["valid"] > 0) & (df["speed"].between(SPEED_MIN, SPEED_MAX))
    return df


def _loadmat_any(path: Path):
    """Deprecated path: kept for backward compatibility if some sessions use MATLAB .mat.

    Prefer MClust-style binary .t/.t64 via scripts.mclust. This helper will only
    be used as a last resort.
    """
    if _scipy_loadmat is not None:
        try:
            return _scipy_loadmat(path)
        except NotImplementedError:
            pass
        except Exception:
            pass
    if _h5py is not None:
        with _h5py.File(path, "r") as f:
            return {k: np.array(v) for k, v in f.items()}
    raise RuntimeError("Neither scipy.io.loadmat nor h5py available to read .t64")


def _parse_tetrode_unit_from_name(stem: str) -> Tuple[int, int]:
    """Extract (tetrode, unit) from various filename stems.

    Supports patterns like:
      - '2018-04-05_1_01' -> (1, 1)
      - 'TT3_2' -> (3, 2)
      - generic: last two integer groups in the name
    """
    # date_tet_unit
    m = re.match(r".*_(\d+)_(\d+)$", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    # TT style
    m = re.match(r"[Tt]{2}(\d+)[^\d]*(\d+)$", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    # fallback: take last two digit groups
    nums = re.findall(r"(\d+)", stem)
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    # unknown
    return -1, 1


def load_units_t64(session_dir: Path) -> Dict[Tuple[int, int], np.ndarray]:
    """Load spike times from .t64 files using the local MClust parser.

    This handles Axona-style binary .t64 with a text header followed by binary timestamps.
    Returns a dict mapping (tetrode, unit) -> times in seconds.
    """
    out: Dict[Tuple[int, int], np.ndarray] = {}
    session_dir = Path(session_dir)
    t64s = list(session_dir.glob("*_*.t64")) or list(session_dir.glob("TT*.t64"))
    if not t64s:
        return out
    try:
        from scripts import mclust as _mclust
    except Exception as e:
        # Fallback: try MATLAB/HDF5 if MClust parser isn't available
        for f in t64s:
            try:
                mat = _loadmat_any(f)
            except Exception:
                continue
            k = next((key for key in mat.keys() if str(key).lower().startswith("t")), None)
            if k is None:
                continue
            data = mat[k]
            def to_seconds(ts):
                ts = np.array(ts).squeeze().astype(np.float64)
                return ts / SPK_FS
            tet, unit = _parse_tetrode_unit_from_name(f.stem)
            if getattr(data, "dtype", None) == object:
                arrs = [to_seconds(cell) for cell in np.array(data).squeeze().tolist()]
                for i, ts in enumerate(arrs, start=1):
                    out[(tet if tet != -1 else 0, i)] = ts
            else:
                out[(tet, unit)] = to_seconds(data)
        return out
    # Preferred path: use local MClust loader
    for f in t64s:
        try:
            hdr, ts = _mclust.load_tfile(str(f))
        except Exception:
            # as a last resort, skip this file
            continue
        tet, unit = _parse_tetrode_unit_from_name(f.stem)
        out[(tet, unit)] = np.asarray(ts, dtype=float)  # already seconds in mclust
    return out


def load_units_mclust(session_dir: Path):
    """Optional: use scripts/mclust.py to read cuts if available."""
    try:
        from scripts import mclust as _mclust  # local project loader
    except Exception as e:  # pragma: no cover
        raise ImportError("scripts.mclust not available") from e
    cuts = _mclust.load_all_cuts(Path(session_dir))
    units = {}
    if not cuts:
        return units
    for tetrode, unit_map in cuts.items():
        for unit, (hdr, ts) in unit_map.items():
            units[(int(tetrode), int(unit))] = np.asarray(ts, dtype=float)
    return units
