"""
Synchronize the ROV video with its telemetry log.

The Deep Trekker rig produces:
- combined_log.csv  : dense state (depth, position, orientation, ...)
- depth_log.csv, orientation_log.csv, position/global, position/local
- log.dtlog         : binary; ignored for now

This module aligns frame indices with telemetry rows so that any
downstream component can ask "what depth/coords/heading at frame N?".
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


@dataclass
class SyncedTelemetry:
    df: pd.DataFrame
    fps: float

    def at_frame(self, frame_idx: int) -> pd.Series:
        if frame_idx < 0 or frame_idx >= len(self.df):
            raise IndexError(f"frame_idx {frame_idx} out of range")
        return self.df.iloc[frame_idx]


# Run scripts/explore_logs.py first against your actual combined_log.csv
# to discover the real column names, then adjust this map. Deep Trekker
# firmwares vary slightly in headers.
COLUMN_MAP_DEFAULT: dict[str, str] = {
    'timestamp':         't_iso',
    'vehicle latitude':  'lat',
    'vehicle longitude': 'lon',
    'depth':             'depth_raw',
    'heading (°)':       'heading_deg',
    'roll (°)':          'roll_deg',
    'pitch (°)':         'pitch_deg',
}


def load_combined_log(path: Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load the Deep Trekker combined log and normalize column names + types."""
    df = pd.read_csv(path)
    cmap = column_map if column_map is not None else COLUMN_MAP_DEFAULT
    df = df.rename(columns=cmap)

    # Deep Trekker timestamps look like '2025.05.13 09:49:43:278' — last
    # colon separates milliseconds. Replace with '.' for pandas to parse.
    if 't_iso' in df.columns:
        ts_str = df['t_iso'].astype(str).str.replace(
            r':(\d{3})$', r'.\1', regex=True
        )
        ts = pd.to_datetime(
            ts_str, format='%Y.%m.%d %H:%M:%S.%f', errors='coerce'
        )
        df['t_sec'] = (ts - ts.iloc[0]).dt.total_seconds()

    # Depth is stored as e.g. '0.430M' — strip the unit and convert.
    if 'depth_raw' in df.columns:
        df['depth_m'] = pd.to_numeric(
            df['depth_raw'].astype(str).str.rstrip('Mm '),
            errors='coerce'
        )

    return df


def load_gpx(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag.startswith('{'):
        ns_uri = root.tag.split('}', 1)[0][1:]
        ns = {'g': ns_uri}
        trkpt_q, ele_q, time_q = './/g:trkpt', 'g:ele', 'g:time'
    else:
        ns = {}
        trkpt_q, ele_q, time_q = './/trkpt', 'ele', 'time'

    pts = root.findall(trkpt_q, ns)
    lats: list[float] = []
    lons: list[float] = []
    eles: list[float | None] = []
    times: list[str | None] = []
    for pt in pts:
        lats.append(float(pt.get('lat')))
        lons.append(float(pt.get('lon')))
        ele = pt.find(ele_q, ns)
        eles.append(float(ele.text) if ele is not None and ele.text else None)
        t = pt.find(time_q, ns)
        times.append(t.text if t is not None else None)

    ts = pd.to_datetime(pd.Series(times), errors='coerce', utc=True)
    t_sec = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()

    data: dict[str, object] = {'t_sec': t_sec, 'lat': lats, 'lon': lons}
    if any(e is not None for e in eles):
        data['elevation_m'] = eles
    return pd.DataFrame(data)


def compute_speed(df: pd.DataFrame) -> pd.Series:
    if {'lat', 'lon'}.issubset(df.columns):
        R = 6_371_000.0
        lat_rad = np.radians(df['lat'].to_numpy(dtype=float))
        lon_rad = np.radians(df['lon'].to_numpy(dtype=float))
        dx = R * np.cos(lat_rad[:-1]) * np.diff(lon_rad)
        dy = R * np.diff(lat_rad)
        d = np.sqrt(dx ** 2 + dy ** 2)
    elif {'x', 'y'}.issubset(df.columns):
        d = np.sqrt(np.diff(df['x']) ** 2 + np.diff(df['y']) ** 2)
    else:
        return pd.Series(np.zeros(len(df)), index=df.index, name='speed_mps')

    dt = np.diff(df['t_sec'].to_numpy(dtype=float))
    speed = np.where(dt > 0, d / dt, 0.0)
    return pd.Series(np.concatenate([[0.0], speed]), index=df.index, name='speed_mps')


def synchronize(
    log_path: Path,
    fps: float,
    n_frames: int,
    column_map: dict[str, str] | None = None,
    gpx_path: Path | None = None,
) -> SyncedTelemetry:
    df = load_combined_log(log_path, column_map=column_map)
    if 't_sec' not in df.columns:
        raise KeyError(
            "Need a 't_sec' column. Inspect your log with "
            "scripts/explore_logs.py and adjust COLUMN_MAP_DEFAULT in sync.py."
        )

    df = df.sort_values('t_sec').reset_index(drop=True)
    df['speed_mps'] = compute_speed(df)

    frame_t = np.arange(n_frames) / fps
    out = pd.DataFrame({'frame_idx': np.arange(n_frames), 't_sec': frame_t})

    numeric_cols = [c for c in df.columns if c != 't_sec' and df[c].dtype.kind in 'fi']
    for col in numeric_cols:
        out[col] = np.interp(frame_t, df['t_sec'], df[col].astype(float))

    if gpx_path is not None:
        gpx = load_gpx(gpx_path)
        for col in ('lat', 'lon', 'elevation_m'):
            if col in gpx.columns:
                out[col] = np.interp(frame_t, gpx['t_sec'], gpx[col].astype(float))

    return SyncedTelemetry(df=out, fps=fps)
