"""数据导入：csv / tsv / txt / data / xlsx / xls / json，自动探测编码、分隔符、表头。"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

NA_VALUES = ["?", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "-", "--", ""]
EXCEL_EXTS = {".xlsx", ".xlsm", ".xls"}
TEXT_EXTS = {".csv", ".tsv", ".txt", ".dat", ".data"}
TABLE_EXTS = EXCEL_EXTS | TEXT_EXTS | {".json"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

FILE_FILTER = "数据文件 (*.csv *.tsv *.txt *.dat *.data *.xlsx *.xlsm *.xls *.json);;所有文件 (*)"


def _detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for enc in ("utf-8", "gb18030"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _split_line(line: str, delim: str) -> list[str]:
    if delim == "ws":
        return re.split(r"\s+", line.strip())
    return next(csv.reader([line], delimiter=delim))


def _sniff_delimiter(lines: list[str]) -> str:
    """返回分隔符；'ws' 表示任意空白。"""
    best, best_score = ",", (-1.0, 0)
    for delim in [",", "\t", ";", "|", "ws"]:
        counts = []
        for ln in lines:
            try:
                counts.append(len(_split_line(ln, delim)))
            except csv.Error:
                counts.append(1)
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        if mode < 2:
            continue
        score = (counts.count(mode) / len(counts), mode)
        if score > best_score:
            best, best_score = delim, score
    return best


def _is_number(token: str) -> bool:
    token = token.strip()
    if token in NA_VALUES:
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def list_sheets(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() not in EXCEL_EXTS:
        return []
    return list(pd.ExcelFile(path).sheet_names)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    names, seen = [], {}
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"col_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df = df.copy()
    df.columns = names
    return df.reset_index(drop=True)


def _coerce_numeric_like(df: pd.DataFrame, min_ratio: float = 0.95) -> pd.DataFrame:
    """字符串列中绝大多数能转成数值的，转为数值（其余变 NaN）。"""
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        nn = s.dropna()
        if nn.empty:
            continue
        conv = pd.to_numeric(nn.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
        if conv.notna().mean() >= min_ratio:
            df[c] = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    return df


def load_table(path: str | Path, sheet: str | int | None = None, header: bool | None = None) -> tuple[pd.DataFrame, dict]:
    """读取表格数据。返回 (DataFrame, info)。header=None 表示自动判断。"""
    path = Path(path)
    ext = path.suffix.lower()
    info: dict = {"path": str(path), "file": path.name, "ext": ext, "notes": []}

    if ext in EXCEL_EXTS:
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0,
                           na_values=NA_VALUES, header=0 if header is not False else None)
        info["sheet"] = sheet if sheet is not None else list_sheets(path)[0]
    elif ext == ".json":
        text = path.read_bytes()
        enc = _detect_encoding(text)
        data = json.loads(text.decode(enc))
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        df = pd.json_normalize(data)
        info["encoding"] = enc
    else:
        raw = path.read_bytes()
        enc = _detect_encoding(raw)
        text = raw.decode(enc, errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()][:50]
        if not lines:
            raise ValueError("文件为空")
        delim = _sniff_delimiter(lines)
        info.update(encoding=enc, delimiter={"ws": "空白", "\t": "制表符"}.get(delim, delim))
        first = _split_line(lines[0], delim)
        if header is None:
            has_header = not all(_is_number(t) for t in first)
        else:
            has_header = header
        info["header"] = has_header
        kwargs = dict(na_values=NA_VALUES, encoding=enc, header=0 if has_header else None,
                      skipinitialspace=True)
        if delim == "ws":
            df = pd.read_csv(path, sep=r"\s+", engine="python", **kwargs)
        else:
            df = pd.read_csv(path, sep=delim, **kwargs)
        if not has_header:
            from .medical import HEART_COLUMNS
            if df.shape[1] == len(HEART_COLUMNS):
                df.columns = HEART_COLUMNS
                info["notes"].append("无表头且为 14 列，已按 UCI 心脏病数据集字段命名")
            else:
                df.columns = [f"col_{i + 1}" for i in range(df.shape[1])]
                info["notes"].append("文件无表头，已自动命名为 col_1…")

    df = _clean_columns(df)
    df = _coerce_numeric_like(df)
    info["shape"] = df.shape
    return df, info
