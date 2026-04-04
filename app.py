# app.py
import json
import os
import re
import unicodedata
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "キャットフード分析比較アプリ"
DATA_FILE = "foods.json"

# ---- CSV/表示で使う列名（キーはこの文字列で固定）----
COL_NAME = "商品名"
COL_PRICE = "価格(円)"
COL_CONTENT = "内容量(g)"
COL_MAIN_ING = "主原材料"

COL_CAL = "カロリー(kcal/100g)"
COL_PROTEIN = "タンパク質(g/100g)"
COL_FAT = "脂質(g/100g)"
COL_REVIEW = "口コミ評価(0-5)"
COL_WATER = "水分(%)"
COL_ASH = "灰分(%)"
COL_CARBS = "炭水化物(%)"
COL_PHOSPHORUS_PCT = "リン(%)"

# 乾物換算（比較表示用）
COL_PROTEIN_DRY = "タンパク質(乾物換算%)"
COL_FAT_DRY = "脂質(乾物換算%)"
COL_CARBS_DRY = "炭水化物(乾物換算%)"
COL_PHOSPHORUS_DRY = "リン(乾物換算%)"
COL_CALCIUM_DRY = "カルシウム(乾物換算%)"
COL_MAGNESIUM_DRY = "マグネシウム(乾物換算%)"

# ミネラル類（以後は mg/100g として入力）※以前の g/100g データと互換を取るため、旧列名も保持
COL_PHOSPHORUS_G = "リン(g/100g)"  # 旧: 互換用
COL_PHOSPHORUS = "リン(mg/100g)"  # 新: 入力・保存は mg/100g
COL_CALCIUM = "カルシウム(g/100g)"
COL_MAGNESIUM = "マグネシウム(g/100g)"

# 添加物
COL_PRESERVATIVE = "保存料(種類)"
COL_COLORANT = "着色料(種類)"
COL_ANTIOXIDANT = "酸化防止剤(種類)"

# 原産国
COL_ORIGIN = "原産国"

# 対象猫（年齢・体重）
COL_TARGET_AGE = "対象猫年齢(歳)"
COL_TARGET_WEIGHT = "対象猫体重(kg)"

# フードの種類
COL_FOOD_TYPE = "フードの種類"

# ランキング用: 動物性タンパク質の割合（0〜100%、高いほど良い）
COL_ANIMAL_PROTEIN_PCT = "動物性タンパク質(%)"

COL_PRICE_100G = "100gあたり価格(円)"
COL_SCORE = "総合スコア"

FOOD_TYPE_OPTIONS = ["ウェット", "ドライ", "セミモイスト"]

# CSVの要求列名（指定フォーマットに合わせる）
CSV_COL_ADDITIVES = "添加物"
CSV_COL_TARGET_CAT = "対象猫"

CSV_TEMPLATE_COLS = [
    COL_NAME,
    COL_PRICE,
    COL_CONTENT,
    COL_MAIN_ING,
    COL_CAL,
    COL_PROTEIN,
    COL_FAT,
    COL_PHOSPHORUS,
    COL_CALCIUM,
    COL_MAGNESIUM,
    CSV_COL_ADDITIVES,
    COL_ORIGIN,
    CSV_COL_TARGET_CAT,
    COL_FOOD_TYPE,
    COL_REVIEW,
]

# 旧CSVとの互換のため必須ではない列（テンプレ・エクスポートには含める）
CSV_OPTIONAL_COLS = [COL_ANIMAL_PROTEIN_PCT]

# CSVヘッダの「別名」（単位なし等）を内部列名へ吸収するためのマッピング
# ユーザー要望の列名例:
# 商品名, 価格, 内容量, 主原材料, カロリー, タンパク質, 脂質, リン, カルシウム, マグネシウム, 添加物, 原産国, 対象猫, フードの種類, 口コミ評価
CSV_HEADER_ALIASES = {
    "価格": COL_PRICE,
    "内容量": COL_CONTENT,
    "カロリー": COL_CAL,
    "タンパク質": COL_PROTEIN,
    "脂質": COL_FAT,
    # リンは mg/100g 入力。過去に g/100g を貼ってしまうケースに備えてヘッダ別名も吸収する
    "リン": COL_PHOSPHORUS,
    "リン(mg/100g)": COL_PHOSPHORUS,
    "リン(g/100g)": COL_PHOSPHORUS,  # 後段で値をスケールして補正
    "カルシウム": COL_CALCIUM,
    "マグネシウム": COL_MAGNESIUM,
    "口コミ評価": COL_REVIEW,
    "動物性タンパク質": COL_ANIMAL_PROTEIN_PCT,
}

STORE_COLUMNS = [
    "id",  # 保存用（CSVテンプレには出さない）
    COL_NAME,
    COL_PRICE,
    COL_CONTENT,
    COL_MAIN_ING,
    COL_FOOD_TYPE,
    COL_ORIGIN,
    COL_TARGET_AGE,
    COL_TARGET_WEIGHT,
    COL_CAL,
    COL_PROTEIN,
    COL_FAT,
    COL_WATER,
    COL_ASH,
    COL_PHOSPHORUS,
    COL_CALCIUM,
    COL_MAGNESIUM,
    COL_PRESERVATIVE,
    COL_COLORANT,
    COL_ANTIOXIDANT,
    COL_REVIEW,
    COL_ANIMAL_PROTEIN_PCT,
]

NUMERIC_COLUMNS = [
    COL_PRICE,
    COL_CONTENT,
    COL_CAL,
    COL_PROTEIN,
    COL_FAT,
    COL_WATER,
    COL_ASH,
    COL_PHOSPHORUS,
    COL_CALCIUM,
    COL_MAGNESIUM,
    COL_TARGET_AGE,
    COL_TARGET_WEIGHT,
    COL_REVIEW,
    COL_ANIMAL_PROTEIN_PCT,
]

TEXT_COLUMNS = [
    COL_NAME,
    COL_MAIN_ING,
    COL_FOOD_TYPE,
    COL_ORIGIN,
    COL_PRESERVATIVE,
    COL_COLORANT,
    COL_ANTIOXIDANT,
]


def _data_path() -> str:
    # app.py と同じディレクトリに保存
    return os.path.join(os.path.dirname(__file__), DATA_FILE)


def load_foods() -> List[Dict[str, Any]]:
    path = _data_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        # データ破損時は初期化せず空にして復旧可能にする
        return []
    return []


def save_foods(foods: List[Dict[str, Any]]) -> None:
    path = _data_path()
    # JSONに NaN を残さない（Excel/再読み込みの事故を減らす）
    sanitized: List[Dict[str, Any]] = []
    for item in foods:
        clean: Dict[str, Any] = {}
        for k, v in item.items():
            if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        sanitized.append(clean)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)


def minmax_norm(s: pd.Series, *, higher_is_better: bool) -> pd.Series:
    s = s.astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    s_non_na = s.dropna()

    if len(s) == 0 or s_non_na.empty:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)

    mn = float(s_non_na.min())
    mx = float(s_non_na.max())
    if mx == mn:
        out = pd.Series([0.5] * len(s), index=s.index, dtype=float)
        return out

    norm = (s - mn) / (mx - mn)
    norm = norm.fillna(0.5)
    return norm if higher_is_better else (1 - norm)


def foods_to_df(foods: List[Dict[str, Any]]) -> pd.DataFrame:
    if not foods:
        return pd.DataFrame(
            columns=[
                *STORE_COLUMNS,
                COL_PRICE_100G,
                COL_CARBS,
                COL_PHOSPHORUS_PCT,
                COL_PROTEIN_DRY,
                COL_FAT_DRY,
                COL_CARBS_DRY,
                COL_PHOSPHORUS_DRY,
                COL_CALCIUM_DRY,
                COL_MAGNESIUM_DRY,
            ]
        )
    df = pd.DataFrame(foods)

    # 旧JSON互換: リンが g/100g の場合は mg/100g に変換
    if COL_PHOSPHORUS_G in df.columns and COL_PHOSPHORUS not in df.columns:
        df[COL_PHOSPHORUS] = pd.to_numeric(df[COL_PHOSPHORUS_G], errors="coerce") * 1000.0

    # 古いデータが混じっていても列名を揃える（不足列は欠損として扱う）
    for col in STORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # 数値列は数値化
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # テキスト列は欠損を空文字に
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # 100gあたり価格（価格 / (内容量g/100)）
    denom = df[COL_CONTENT] / 100.0
    denom = denom.replace(0, np.nan)
    df[COL_PRICE_100G] = df[COL_PRICE] / denom

    # 炭水化物(%)を自動計算
    # 計算式: 100 - タンパク質 - 脂質 - 水分 - 灰分
    # 水分・灰分が0の場合は簡易計算: 100 - タンパク質 - 脂質
    water = df[COL_WATER].fillna(0.0)
    ash = df[COL_ASH].fillna(0.0)
    carbs_full = 100.0 - df[COL_PROTEIN] - df[COL_FAT] - water - ash
    carbs_simple = 100.0 - df[COL_PROTEIN] - df[COL_FAT]
    use_simple = (water == 0.0) & (ash == 0.0)
    df[COL_CARBS] = carbs_full.where(~use_simple, carbs_simple)

    # リンの表示用: mg/100g -> %（= g/100g）
    df[COL_PHOSPHORUS_PCT] = df[COL_PHOSPHORUS] / 1000.0

    # 乾物換算（ウェットのみ）: 栄養素(%) ÷ (100 - 水分%) × 100
    # ここで扱うのは「%としての栄養素」（タンパク質・脂質・炭水化物・リン(%)・カルシウム・マグネシウム）
    water_pct = df[COL_WATER].fillna(0.0)
    denom = 100.0 - water_pct
    factor = (100.0 / denom).where(denom > 0, np.nan)
    is_wet = df[COL_FOOD_TYPE] == "ウェット"

    df[COL_PROTEIN_DRY] = df[COL_PROTEIN].where(~is_wet, df[COL_PROTEIN] * factor)
    df[COL_FAT_DRY] = df[COL_FAT].where(~is_wet, df[COL_FAT] * factor)
    df[COL_CARBS_DRY] = df[COL_CARBS].where(~is_wet, df[COL_CARBS] * factor)
    df[COL_PHOSPHORUS_DRY] = df[COL_PHOSPHORUS_PCT].where(~is_wet, df[COL_PHOSPHORUS_PCT] * factor)
    df[COL_CALCIUM_DRY] = df[COL_CALCIUM].where(~is_wet, df[COL_CALCIUM] * factor)
    df[COL_MAGNESIUM_DRY] = df[COL_MAGNESIUM].where(~is_wet, df[COL_MAGNESIUM] * factor)
    return df


def _coerce_numeric_from_csv(s: pd.Series) -> pd.Series:
    # 例: "1,234" を扱うためカンマを除去
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _coerce_text_from_csv(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
    return s.fillna("").astype(str)


def csv_template_bytes() -> bytes:
    # CSVテンプレはユーザー要望の列 + 任意列（指定順）
    csv_text = ",".join([*CSV_TEMPLATE_COLS, *CSV_OPTIONAL_COLS]) + "\n"
    # Excelでの文字化け対策
    return csv_text.encode("utf-8-sig")


def foods_from_csv(file_bytes: bytes, *, mode: str = "append", debug: bool = False) -> List[Dict[str, Any]]:
    # mode は現状未使用（呼び側で追記/上書きの制御をする想定）
    from io import BytesIO

    def normalize_header_key(s: Any) -> str:
        """
        CSVヘッダの表記ゆれ吸収用（空白除去、全角/半角・括弧差などを正規化）
        """
        s2 = unicodedata.normalize("NFKC", str(s))
        s2 = s2.strip()
        s2 = s2.replace("（", "(").replace("）", ")")
        s2 = re.sub(r"\s+", "", s2)
        return s2

    # デバッグ用: CSVの1行目ヘッダー（生の文字列）を可能な限りそのまま表示する
    raw_header_line: str = ""
    newline_idx = file_bytes.find(b"\n")
    if newline_idx == -1:
        header_bytes = file_bytes
    else:
        header_bytes = file_bytes[:newline_idx].rstrip(b"\r")

    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "sjis", "latin-1"]:
        try:
            raw_header_line = header_bytes.decode(enc)
            break
        except Exception:
            continue

    # 文字コードが不明なCSVをある程度吸収する
    last_err: Optional[Exception] = None
    df_csv: Optional[pd.DataFrame] = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "sjis", "latin-1"]:
        try:
            df_csv = pd.read_csv(BytesIO(file_bytes), dtype=str, encoding=enc)
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    if df_csv is None:
        raise ValueError(f"CSVの読み込みに失敗しました: {last_err}")

    df_csv.columns = [str(c).strip() for c in df_csv.columns]

    # リンの単位だけ（旧: g/100g -> 新: mg/100g）を値側で補正する
    phos_g_present = any(
        normalize_header_key(c) == normalize_header_key(COL_PHOSPHORUS_G) for c in df_csv.columns
    )
    phos_mg_present = any(
        normalize_header_key(c) == normalize_header_key(COL_PHOSPHORUS) for c in df_csv.columns
    ) or any(normalize_header_key(c) == normalize_header_key("リン") for c in df_csv.columns)

    # デバッグ表示（ヘッダー/認識列/エイリアス適用後）
    if debug:
        with st.expander("CSVデバッグ情報（インポート時）", expanded=True):
            st.write("CSV 1行目ヘッダー（生データ）:")
            st.code(raw_header_line, language="text")
            st.write("pandas が認識した列名（リネーム前）:")
            st.code(", ".join(list(df_csv.columns)), language="text")

    # ヘッダ別名（単位なしなど）を内部列名へリネーム（表記ゆれも吸収）
    alias_norm_map: Dict[str, str] = {
        normalize_header_key(k): v for k, v in CSV_HEADER_ALIASES.items()
    }
    exact_norm_map: Dict[str, str] = {
        normalize_header_key(v): v for v in [*CSV_TEMPLATE_COLS, *CSV_OPTIONAL_COLS]
    }

    rename_map: Dict[str, str] = {}
    for col in df_csv.columns:
        col_s = str(col).strip()
        col_norm = normalize_header_key(col_s)
        if col_norm in alias_norm_map:
            rename_map[col_s] = alias_norm_map[col_norm]
        elif col_norm in exact_norm_map and col_s != exact_norm_map[col_norm]:
            # 例: "価格（円）" -> "価格(円)" のような表記ゆれ吸収
            rename_map[col_s] = exact_norm_map[col_norm]

    if rename_map:
        df_csv = df_csv.rename(columns=rename_map)
        df_csv.columns = [str(c).strip() for c in df_csv.columns]
        # 同名列が複数できてしまうケースを避ける
        df_csv = df_csv.loc[:, ~pd.Index(df_csv.columns).duplicated()]

        # リン（旧g/100g入力）を mg/100g に換算（CSVの値側だけ補正）
        if phos_g_present and not phos_mg_present and COL_PHOSPHORUS in df_csv.columns:
            # dtype=str のままだと掛け算できないため数値化してから補正
            phos_numeric = _coerce_numeric_from_csv(df_csv[COL_PHOSPHORUS])
            df_csv[COL_PHOSPHORUS] = phos_numeric * 1000.0

    if debug:
        with st.expander("CSVデバッグ情報（エイリアス適用後）", expanded=False):
            st.write("適用した別名マッピング:")
            st.code(json.dumps(rename_map, ensure_ascii=False, indent=2), language="text")
            st.write("リネーム後の列名:")
            st.code(", ".join(list(df_csv.columns)), language="text")

    required = CSV_TEMPLATE_COLS
    missing = [c for c in required if c not in df_csv.columns]
    if missing:
        raise ValueError(f"CSVに必須列が不足しています: {missing}")

    # 行ごとに辞書へ（指定テンプレ列 -> 内部保存列へ変換）
    foods: List[Dict[str, Any]] = []
    for _, row in df_csv.iterrows():
        name = str(row.get(COL_NAME, "")).strip()
        if not name:
            continue

        def num(col: str) -> Optional[float]:
            v = row.get(col, None)
            if v is None or pd.isna(v):
                return None
            try:
                return float(str(v).replace(",", "").strip())
            except Exception:
                return None

        additives_raw = row.get(CSV_COL_ADDITIVES, None)
        additives_str = "" if additives_raw is None or pd.isna(additives_raw) else str(additives_raw).strip()

        # "添加物"はCSV上は1列だが、入力例に合わせて「保存料/着色料/酸化防止剤」に分解する
        # 例: "保存料:〇〇; 着色料:△△; 酸化防止剤:□□"
        preservative_val = ""
        colorant_val = ""
        antioxidant_val = ""
        for key, val in re.findall(r"(保存料|着色料|酸化防止剤)\s*[:：]\s*([^;；\r\n]+)", additives_str):
            parsed = str(val).strip()
            if key == "保存料":
                preservative_val = parsed
            elif key == "着色料":
                colorant_val = parsed
            elif key == "酸化防止剤":
                antioxidant_val = parsed

        # キーが見つからなければ、旧来の仕様として全体を「保存料」として扱う
        if not any([preservative_val, colorant_val, antioxidant_val]) and additives_str:
            preservative_val = additives_str

        target_raw = row.get(CSV_COL_TARGET_CAT, None)
        target_str = "" if target_raw is None or pd.isna(target_raw) else str(target_raw).strip()
        # 表記ゆれに対応（「才」「Kg」「キロ」「年齢/体重」など）
        age_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:歳|才)", target_str)
        weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|KG|Kg|キロ)", target_str, flags=re.IGNORECASE)
        age = float(age_match.group(1)) if age_match else None
        weight = float(weight_match.group(1)) if weight_match else None

        item: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            COL_NAME: name,
            COL_PRICE: num(COL_PRICE),
            COL_CONTENT: num(COL_CONTENT),
            COL_MAIN_ING: str(row.get(COL_MAIN_ING, "") or "").strip(),
            COL_FOOD_TYPE: str(row.get(COL_FOOD_TYPE, "") or "").strip(),
            COL_ORIGIN: str(row.get(COL_ORIGIN, "") or "").strip(),
            COL_TARGET_AGE: age,
            COL_TARGET_WEIGHT: weight,
            COL_CAL: num(COL_CAL),
            COL_PROTEIN: num(COL_PROTEIN),
            COL_FAT: num(COL_FAT),
            COL_PHOSPHORUS: num(COL_PHOSPHORUS),
            COL_CALCIUM: num(COL_CALCIUM),
            COL_MAGNESIUM: num(COL_MAGNESIUM),
            # "添加物"（1列） -> 内部の3項目のうち保存料に格納（他2つは空）
            COL_PRESERVATIVE: preservative_val,
            COL_COLORANT: colorant_val,
            COL_ANTIOXIDANT: antioxidant_val,
            COL_REVIEW: num(COL_REVIEW),
            COL_ANIMAL_PROTEIN_PCT: num(COL_ANIMAL_PROTEIN_PCT)
            if COL_ANIMAL_PROTEIN_PCT in df_csv.columns
            else None,
        }
        foods.append(item)
    return foods


def foods_to_csv_bytes(foods: List[Dict[str, Any]]) -> bytes:
    df = foods_to_df(foods)
    # "添加物"（1列） / "対象猫"（1列）にまとめて出力（テンプレ指定列と同じ構成）
    def additives_out(row: pd.Series) -> str:
        parts: List[str] = []
        if isinstance(row.get(COL_PRESERVATIVE, ""), str) and row.get(COL_PRESERVATIVE, "").strip():
            parts.append(f"保存料:{row[COL_PRESERVATIVE]}")
        if isinstance(row.get(COL_COLORANT, ""), str) and row.get(COL_COLORANT, "").strip():
            parts.append(f"着色料:{row[COL_COLORANT]}")
        if isinstance(row.get(COL_ANTIOXIDANT, ""), str) and row.get(COL_ANTIOXIDANT, "").strip():
            parts.append(f"酸化防止剤:{row[COL_ANTIOXIDANT]}")
        return "; ".join(parts)

    def target_cat_out(row: pd.Series) -> str:
        parts: List[str] = []
        age = row.get(COL_TARGET_AGE, np.nan)
        weight = row.get(COL_TARGET_WEIGHT, np.nan)
        if pd.notna(age):
            parts.append(f"年齢{age:g}歳")
        if pd.notna(weight):
            parts.append(f"体重{weight:g}kg")
        return ", ".join(parts)

    df_out = pd.DataFrame(
        {
            COL_NAME: df[COL_NAME],
            COL_PRICE: df[COL_PRICE],
            COL_CONTENT: df[COL_CONTENT],
            COL_MAIN_ING: df[COL_MAIN_ING],
            COL_CAL: df[COL_CAL],
            COL_PROTEIN: df[COL_PROTEIN],
            COL_FAT: df[COL_FAT],
            COL_PHOSPHORUS: df[COL_PHOSPHORUS],
            COL_CALCIUM: df[COL_CALCIUM],
            COL_MAGNESIUM: df[COL_MAGNESIUM],
            CSV_COL_ADDITIVES: df.apply(additives_out, axis=1),
            COL_ORIGIN: df[COL_ORIGIN],
            CSV_COL_TARGET_CAT: df.apply(target_cat_out, axis=1),
            COL_FOOD_TYPE: df[COL_FOOD_TYPE],
            COL_REVIEW: df[COL_REVIEW],
            COL_ANIMAL_PROTEIN_PCT: df[COL_ANIMAL_PROTEIN_PCT],
        }
    )
    # Excel/日本語文字化け対策
    export_cols = [*CSV_TEMPLATE_COLS, *CSV_OPTIONAL_COLS]
    return df_out[export_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def _existing_name_set(foods: List[Dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for f in foods:
        n = str(f.get(COL_NAME, "")).strip()
        if n:
            names.add(n)
    return names


def _dedupe_items_by_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同一商品名の重複（同じCSV内）を 1 件にまとめます（最後に出現した行を採用）。
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for item in items:
        n = str(item.get(COL_NAME, "")).strip()
        if not n:
            continue
        by_name[n] = item
    return list(by_name.values())


def _duplicate_names(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[str]:
    existing_names = _existing_name_set(existing)
    names: set[str] = set()
    for item in new_items:
        n = str(item.get(COL_NAME, "")).strip()
        if n and n in existing_names:
            names.add(n)
    return sorted(names)


def compute_ranking(
    df: pd.DataFrame,
    *,
    w_review: float,
    w_protein: float,
    w_cost_100g: float,
    w_fat: float,
    w_calories: float,
    w_phosphorus: float,
    w_carbs: float,
    w_animal_protein: float,
    w_additive_free: float,
) -> pd.DataFrame:
    # 重みが全部0の場合の保険
    weight_sum = (
        w_review
        + w_protein
        + w_cost_100g
        + w_fat
        + w_calories
        + w_phosphorus
        + w_carbs
        + w_animal_protein
        + w_additive_free
    )
    if weight_sum <= 0:
        w_review, w_protein, w_cost_100g, w_fat, w_calories = 0.25, 0.15, 0.15, 0.05, 0.05
        w_phosphorus, w_carbs, w_animal_protein, w_additive_free = 0.1, 0.1, 0.1, 0.1
        weight_sum = 1.0

    w_review /= weight_sum
    w_protein /= weight_sum
    w_cost_100g /= weight_sum
    w_fat /= weight_sum
    w_calories /= weight_sum
    w_phosphorus /= weight_sum
    w_carbs /= weight_sum
    w_animal_protein /= weight_sum
    w_additive_free /= weight_sum

    protein_norm = minmax_norm(df[COL_PROTEIN], higher_is_better=True)
    fat_norm = minmax_norm(df[COL_FAT], higher_is_better=False)  # 少ない方が良い想定
    cal_norm = minmax_norm(df[COL_CAL], higher_is_better=False)  # 少ない方が良い想定
    cost_norm = minmax_norm(df[COL_PRICE_100G], higher_is_better=False)  # 安い方が良い想定
    review_norm = minmax_norm(df[COL_REVIEW], higher_is_better=True)

    phos_norm = minmax_norm(df[COL_PHOSPHORUS], higher_is_better=False)
    carbs_norm = minmax_norm(df[COL_CARBS], higher_is_better=False)
    animal_norm = minmax_norm(df[COL_ANIMAL_PROTEIN_PCT], higher_is_better=True)

    pres = df[COL_PRESERVATIVE].fillna("").astype(str).str.strip()
    colr = df[COL_COLORANT].fillna("").astype(str).str.strip()
    ant = df[COL_ANTIOXIDANT].fillna("").astype(str).str.strip()
    additive_free = ((pres == "") & (colr == "") & (ant == "")).astype(float)
    additive_norm = minmax_norm(additive_free, higher_is_better=True)

    df = df.copy()
    df[COL_SCORE] = (
        w_review * review_norm
        + w_protein * protein_norm
        + w_cost_100g * cost_norm
        + w_fat * fat_norm
        + w_calories * cal_norm
        + w_phosphorus * phos_norm
        + w_carbs * carbs_norm
        + w_animal_protein * animal_norm
        + w_additive_free * additive_norm
    )
    return df


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("キャットフードを登録し、栄養素・コスパ・総合スコアで比較できます。")

    if "foods" not in st.session_state:
        st.session_state.foods = load_foods()

    foods: List[Dict[str, Any]] = st.session_state.foods
    df = foods_to_df(foods)

    # ---- メイン画面上部にCSV操作ボタンを追加（テンプレDL / インポート / エクスポート）----
    with st.container():
        st.subheader("CSV一括操作")
        c_template, c_import, c_export = st.columns(3)

        with c_template:
            st.download_button(
                label="CSVテンプレートをダウンロード",
                data=csv_template_bytes(),
                file_name="catfood_template.csv",
                mime="text/csv",
                use_container_width=True,
                key="top_csv_template_dl",
            )

        with c_import:
            uploaded_top = st.file_uploader("CSVファイルを選択（追記インポート）", type=["csv"], key="top_csv_uploader")

            dup_mode_top = st.radio(
                "既存データとの重複時",
                ["スキップ", "上書き（確認）"],
                index=0,
                horizontal=True,
                key="dup_mode_top",
            )

            if st.button(
                "CSVを追記インポート",
                type="primary",
                disabled=(uploaded_top is None),
                key="top_csv_import_btn",
            ):
                try:
                    file_bytes = uploaded_top.read()
                    new_items = _dedupe_items_by_name(foods_from_csv(file_bytes, debug=True))
                    if not new_items:
                        st.warning("商品名が空の行はスキップされました。追加件数がありません。")
                    else:
                        existing = st.session_state.foods
                        existing_names = _existing_name_set(existing)
                        dup_names = _duplicate_names(existing, new_items)

                        if dup_names and dup_mode_top == "スキップ":
                            filtered = [
                                it for it in new_items
                                if str(it.get(COL_NAME, "")).strip() and str(it.get(COL_NAME, "")).strip() not in existing_names
                            ]
                            if not filtered:
                                st.warning(f"重複のため追加できる商品がありません。({len(dup_names)}件はスキップ)")
                            else:
                                st.session_state.foods.extend(filtered)
                                save_foods(st.session_state.foods)
                                st.success(f"インポート完了: {len(filtered)}件追加しました。（重複 {len(dup_names)}件はスキップ）")
                                st.rerun()
                        elif dup_names and dup_mode_top == "上書き（確認）":
                            st.session_state["csv_import_pending_overwrite_top"] = {
                                "new_items": new_items,
                                "dup_names": dup_names,
                            }
                            st.warning(f"重複商品名が {len(dup_names)} 件あります。上書きを実行するには確認してください。")
                        else:
                            st.session_state.foods.extend(new_items)
                            save_foods(st.session_state.foods)
                            st.success(f"インポート完了: {len(new_items)}件追加しました。")
                            st.rerun()
                except Exception as e:
                    st.error(f"インポートに失敗しました: {e}")

            pending = st.session_state.get("csv_import_pending_overwrite_top")
            if pending and isinstance(pending, dict):
                dup_names = pending.get("dup_names", [])
                st.info(f"上書き確認（重複: {len(dup_names)}件）")
                if st.button("重複を上書きして反映（既存を置換）", type="primary", key="top_csv_overwrite_confirm"):
                    dup_set = set(dup_names)
                    # 既存の重複分を削除して、新データを追加
                    st.session_state.foods = [
                        f for f in st.session_state.foods
                        if str(f.get(COL_NAME, "")).strip() not in dup_set
                    ] + pending.get("new_items", [])
                    save_foods(st.session_state.foods)
                    st.success("上書きインポートを完了しました。")
                    st.session_state.pop("csv_import_pending_overwrite_top", None)
                    st.rerun()

                if st.button("キャンセル", key="top_csv_overwrite_cancel"):
                    st.session_state.pop("csv_import_pending_overwrite_top", None)
                    st.rerun()

        with c_export:
            csv_bytes = foods_to_csv_bytes(foods)
            st.download_button(
                label="現在データをCSVでエクスポート",
                data=csv_bytes,
                file_name="catfood_export.csv",
                mime="text/csv",
                use_container_width=True,
                key="top_csv_export_dl",
            )

    with st.sidebar:
        st.header("ランキング重み（調整）")
        w_review = st.slider("口コミ評価", 0.0, 1.0, 0.25, 0.05)
        w_protein = st.slider("タンパク質", 0.0, 1.0, 0.15, 0.05)
        w_cost_100g = st.slider("100gあたり価格（安いほど）", 0.0, 1.0, 0.15, 0.05)
        w_fat = st.slider("脂質（少ないほど）", 0.0, 1.0, 0.05, 0.05)
        w_calories = st.slider("カロリー（少ないほど）", 0.0, 1.0, 0.05, 0.05)
        w_phosphorus = st.slider("リン（少ないほど）", 0.0, 1.0, 0.1, 0.05)
        w_carbs = st.slider("炭水化物（少ないほど）", 0.0, 1.0, 0.1, 0.05)
        w_animal_protein = st.slider("動物性タンパク質（多いほど）", 0.0, 1.0, 0.1, 0.05)
        w_additive_free = st.slider("添加物なし（無添加ほど）", 0.0, 1.0, 0.1, 0.05)

        st.divider()
        st.subheader("データ")
        st.write(f"登録数: {len(foods)}")
        st.info("総合スコアは登録された商品の中での相対比較（正規化）です。")

    tab_register, tab_list, tab_compare, tab_rank = st.tabs(["登録", "一覧", "比較", "ランキング"])

    with tab_register:
        st.subheader("キャットフードを登録")
        with st.form("food_form", clear_on_submit=True):
            name = st.text_input("商品名", max_chars=80)
            price = st.number_input("価格(円)", min_value=0.0, step=1.0, format="%.0f")
            content_g = st.number_input("内容量(g)", min_value=1.0, step=1.0, format="%.0f")
            main_ing = st.text_input("主原材料", max_chars=120)

            food_type = st.selectbox(COL_FOOD_TYPE, options=FOOD_TYPE_OPTIONS, index=0)
            origin = st.text_input(COL_ORIGIN, max_chars=80)
            target_age = st.number_input(COL_TARGET_AGE, min_value=0.0, step=0.1, format="%.1f")
            target_weight = st.number_input(COL_TARGET_WEIGHT, min_value=0.0, step=0.1, format="%.1f")

            calories = st.number_input("カロリー(kcal/100g)", min_value=0.0, step=1.0)
            protein = st.number_input("タンパク質(g/100g)", min_value=0.0, step=0.1)
            fat = st.number_input("脂質(g/100g)", min_value=0.0, step=0.1)

            water = st.number_input(COL_WATER, min_value=0.0, step=0.1, format="%.1f")
            ash = st.number_input(COL_ASH, min_value=0.0, step=0.1, format="%.1f")

            # 炭水化物(%)を自動計算して表示
            if float(water) == 0.0 and float(ash) == 0.0:
                carbs_calc = 100.0 - float(protein) - float(fat)
            else:
                carbs_calc = 100.0 - float(protein) - float(fat) - float(water) - float(ash)
            st.metric("炭水化物(%)", f"{carbs_calc:.1f}%")
            if carbs_calc >= 30.0:
                st.markdown(
                    f"<span style='color: red; font-weight: bold;'>炭水化物が高めで {carbs_calc:.1f}% です（30%以上）</span>",
                    unsafe_allow_html=True,
                )

            phosphorus = st.number_input(COL_PHOSPHORUS, min_value=0.0, step=1.0, format="%.0f")
            # リン(%): mg/100g -> %（= g/100g）へ換算
            st.metric(COL_PHOSPHORUS_PCT, f"{(phosphorus / 1000.0):.3f}%")
            calcium = st.number_input(COL_CALCIUM, min_value=0.0, step=0.1, format="%.2f")
            magnesium = st.number_input(COL_MAGNESIUM, min_value=0.0, step=0.1, format="%.2f")

            # ウェットのみ: 乾物換算値を表示（ドライフードと比較しやすくするため）
            if food_type == "ウェット":
                denom = 100.0 - float(water)
                if denom > 0:
                    protein_dry_calc = float(protein) / denom * 100.0
                    fat_dry_calc = float(fat) / denom * 100.0
                    carbs_dry_calc = float(carbs_calc) / denom * 100.0
                    phosphorus_pct = float(phosphorus) / 1000.0
                    phosphorus_dry_calc = phosphorus_pct / denom * 100.0
                    calcium_dry_calc = float(calcium) / denom * 100.0
                    magnesium_dry_calc = float(magnesium) / denom * 100.0

                    st.info(
                        "乾物換算は、水分を除いた基準で栄養素を換算するため、ウェット（高水分）でもドライと公平に比較できます。"
                    )
                    with st.expander("乾物換算（ウェットのみ）", expanded=False):
                        st.metric(COL_PROTEIN_DRY, f"{protein_dry_calc:.2f}%")
                        st.metric(COL_FAT_DRY, f"{fat_dry_calc:.2f}%")
                        st.metric(COL_CARBS_DRY, f"{carbs_dry_calc:.1f}%")
                        st.metric(COL_PHOSPHORUS_DRY, f"{phosphorus_dry_calc:.4f}%")
                        st.metric(COL_CALCIUM_DRY, f"{calcium_dry_calc:.2f}%")
                        st.metric(COL_MAGNESIUM_DRY, f"{magnesium_dry_calc:.2f}%")
                else:
                    st.warning("水分(%)が100に近く、乾物換算ができません。")

            review = st.slider("口コミ評価(0-5)", min_value=0.0, max_value=5.0, step=0.1, value=3.0)

            animal_protein_pct = st.slider(
                COL_ANIMAL_PROTEIN_PCT,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                value=50.0,
                help="タンパク質のうち動物性由来の割合（目安）。不明なら中央値のままでも比較できます。",
            )

            preservative = st.text_input(COL_PRESERVATIVE, max_chars=120)
            colorant = st.text_input(COL_COLORANT, max_chars=120)
            antioxidant = st.text_input(COL_ANTIOXIDANT, max_chars=120)

            submitted = st.form_submit_button("登録")

        if submitted:
            if not name.strip():
                st.error("商品名を入力してください。")
            else:
                new_food: Dict[str, Any] = {
                    "id": str(uuid.uuid4()),
                    COL_NAME: name.strip(),
                    COL_PRICE: float(price),
                    COL_CONTENT: float(content_g),
                    COL_MAIN_ING: main_ing.strip(),
                    COL_FOOD_TYPE: food_type,
                    COL_ORIGIN: origin.strip(),
                    COL_TARGET_AGE: float(target_age),
                    COL_TARGET_WEIGHT: float(target_weight),
                    COL_CAL: float(calories),
                    COL_PROTEIN: float(protein),
                    COL_FAT: float(fat),
                    COL_WATER: float(water),
                    COL_ASH: float(ash),
                    COL_PHOSPHORUS: float(phosphorus),
                    COL_CALCIUM: float(calcium),
                    COL_MAGNESIUM: float(magnesium),
                    COL_PRESERVATIVE: preservative.strip(),
                    COL_COLORANT: colorant.strip(),
                    COL_ANTIOXIDANT: antioxidant.strip(),
                    COL_REVIEW: float(review),
                    COL_ANIMAL_PROTEIN_PCT: float(animal_protein_pct),
                }
                st.session_state.foods.append(new_food)
                save_foods(st.session_state.foods)
                st.success("登録しました。")
                st.rerun()

        st.caption("入力値は JSON ファイル（`foods.json`）に保存されます。")

    with tab_list:
        st.subheader("登録済み一覧")

        if df.empty:
            st.info("まだ登録がありません。`登録` タブで追加してください。")
        else:
            show_cols = [
                COL_NAME,
                COL_PRICE,
                COL_CONTENT,
                COL_PRICE_100G,
                COL_FOOD_TYPE,
                COL_ORIGIN,
                COL_TARGET_AGE,
                COL_TARGET_WEIGHT,
                COL_MAIN_ING,
                COL_CAL,
                COL_PROTEIN,
                COL_FAT,
                COL_CARBS,
                COL_PHOSPHORUS,
                COL_PHOSPHORUS_PCT,
                COL_CALCIUM,
                COL_MAGNESIUM,
                COL_PRESERVATIVE,
                COL_COLORANT,
                COL_ANTIOXIDANT,
                COL_REVIEW,
                COL_ANIMAL_PROTEIN_PCT,
            ]
            df_show = df[show_cols].copy()
            df_show[COL_PRICE] = df_show[COL_PRICE].round(0)
            df_show[COL_CONTENT] = df_show[COL_CONTENT].round(0)
            df_show[COL_PRICE_100G] = df_show[COL_PRICE_100G].round(2)
            df_show[COL_PROTEIN] = df_show[COL_PROTEIN].round(2)
            df_show[COL_FAT] = df_show[COL_FAT].round(2)
            df_show[COL_CARBS] = df_show[COL_CARBS].round(1)
            df_show[COL_CAL] = df_show[COL_CAL].round(1)
            df_show[COL_PHOSPHORUS] = df_show[COL_PHOSPHORUS].round(2)
            df_show[COL_PHOSPHORUS_PCT] = df_show[COL_PHOSPHORUS_PCT].round(4)
            df_show[COL_CALCIUM] = df_show[COL_CALCIUM].round(2)
            df_show[COL_MAGNESIUM] = df_show[COL_MAGNESIUM].round(2)
            df_show[COL_TARGET_AGE] = df_show[COL_TARGET_AGE].round(1)
            df_show[COL_TARGET_WEIGHT] = df_show[COL_TARGET_WEIGHT].round(1)
            df_show[COL_REVIEW] = df_show[COL_REVIEW].round(2)
            df_show[COL_ANIMAL_PROTEIN_PCT] = df_show[COL_ANIMAL_PROTEIN_PCT].round(1)

            st.dataframe(df_show, use_container_width=True, hide_index=True)

            # 炭水化物が高めのデータは赤字で警告表示
            if COL_CARBS in df.columns:
                high_carbs = df[df[COL_CARBS] >= 30.0][[COL_NAME, COL_CARBS]].copy()
                if not high_carbs.empty:
                    st.markdown("### 警告（炭水化物が高い商品）")
                    for _, r in high_carbs.iterrows():
                        st.markdown(
                            f"<span style='color: red; font-weight: bold;'>{r[COL_NAME]}：炭水化物が高めで {float(r[COL_CARBS]):.1f}% です（30%以上）</span>",
                            unsafe_allow_html=True,
                        )

            st.divider()
            tab_csv_import, tab_csv_export = st.tabs(["CSV一括インポート", "CSVエクスポート"])

            with tab_csv_import:
                st.subheader("CSVテンプレート")
                st.download_button(
                    label="CSVテンプレートをダウンロード",
                    data=csv_template_bytes(),
                    file_name="catfood_template.csv",
                    mime="text/csv",
                    key="tab_csv_template_dl",
                )

                with st.expander("テンプレの書き方例（対象猫 / 添加物）", expanded=False):
                    st.write("`対象猫`は、`◯歳` と `◯kg` が分かる形式ならOKです（例: `2歳, 4kg` / `年齢2歳 体重4kg`）。")
                    st.write("`添加物`は1列ですが、次のようにキー付きで書くと3項目に分解されます。")
                    st.code(
                        "保存料:ローズマリー抽出物; 着色料:カラメル; 酸化防止剤:トコフェロール（混合）",
                        language="text",
                    )
                    st.write("上記のキーを省略して `トコフェロール（混合）` のように書いた場合は、保存料として扱います。")
                    st.write(
                        f"テンプレ末尾の `{COL_ANIMAL_PROTEIN_PCT}` は任意です。列が無い古いCSVでもインポートできます。"
                    )

                st.subheader("CSVをアップロードして一括登録")
                uploaded = st.file_uploader("CSVファイルを選択", type=["csv"], key="tab_csv_uploader")

                dup_mode_tab = st.radio(
                    "既存データとの重複時",
                    ["スキップ", "上書き（確認）"],
                    index=0,
                    horizontal=True,
                    key="dup_mode_tab",
                )

                if uploaded is not None:
                    st.caption(f"選択中: {uploaded.name}")

                if st.button(
                    "CSVを追記インポート",
                    type="primary",
                    disabled=(uploaded is None),
                    key="tab_csv_import_btn",
                ):
                    try:
                        file_bytes = uploaded.read()
                        new_items = _dedupe_items_by_name(foods_from_csv(file_bytes, debug=True))
                        if not new_items:
                            st.warning("商品名が空の行はスキップされました。追加件数がありません。")
                        else:
                            existing = st.session_state.foods
                            existing_names = _existing_name_set(existing)
                            dup_names = _duplicate_names(existing, new_items)

                            if dup_names and dup_mode_tab == "スキップ":
                                filtered = [
                                    it for it in new_items
                                    if str(it.get(COL_NAME, "")).strip() and str(it.get(COL_NAME, "")).strip() not in existing_names
                                ]
                                if not filtered:
                                    st.warning(f"重複のため追加できる商品がありません。({len(dup_names)}件はスキップ)")
                                else:
                                    st.session_state.foods.extend(filtered)
                                    save_foods(st.session_state.foods)
                                    st.success(
                                        f"インポート完了: {len(filtered)}件追加しました。（重複 {len(dup_names)}件はスキップ）"
                                    )
                                    st.rerun()
                            elif dup_names and dup_mode_tab == "上書き（確認）":
                                st.session_state["csv_import_pending_overwrite_tab"] = {
                                    "new_items": new_items,
                                    "dup_names": dup_names,
                                }
                                st.warning(f"重複商品名が {len(dup_names)} 件あります。上書きを実行するには確認してください。")
                            else:
                                st.session_state.foods.extend(new_items)
                                save_foods(st.session_state.foods)
                                st.success(f"インポート完了: {len(new_items)}件追加しました。")
                                st.rerun()
                    except Exception as e:
                        st.error(f"インポートに失敗しました: {e}")

                pending = st.session_state.get("csv_import_pending_overwrite_tab")
                if pending and isinstance(pending, dict):
                    dup_names = pending.get("dup_names", [])
                    st.info(f"上書き確認（重複: {len(dup_names)}件）")
                    if st.button(
                        "重複を上書きして反映（既存を置換）",
                        type="primary",
                        key="tab_csv_overwrite_confirm",
                    ):
                        dup_set = set(dup_names)
                        st.session_state.foods = [
                            f for f in st.session_state.foods
                            if str(f.get(COL_NAME, "")).strip() not in dup_set
                        ] + pending.get("new_items", [])
                        save_foods(st.session_state.foods)
                        st.success("上書きインポートを完了しました。")
                        st.session_state.pop("csv_import_pending_overwrite_tab", None)
                        st.rerun()

                    if st.button("キャンセル", key="tab_csv_overwrite_cancel"):
                        st.session_state.pop("csv_import_pending_overwrite_tab", None)
                        st.rerun()

            with tab_csv_export:
                st.subheader("CSVでダウンロード")
                csv_bytes = foods_to_csv_bytes(foods)
                st.download_button(
                    label="登録済みデータをCSVでダウンロード",
                    data=csv_bytes,
                    file_name="catfood_export.csv",
                    mime="text/csv",
                    key="tab_csv_export_dl",
                )

            st.divider()
            st.subheader("重複データの整理")
            # 同一商品名の重複（既存データ）
            name_counts: Dict[str, int] = {}
            for f in foods:
                n = str(f.get(COL_NAME, "")).strip()
                if n:
                    name_counts[n] = name_counts.get(n, 0) + 1
            dup_names_list = [n for n, c in name_counts.items() if c > 1]
            if not dup_names_list:
                st.info("重複はありません。")
            else:
                st.warning(f"重複商品名: {len(dup_names_list)}件（同名を1件だけ残します）")
                if not st.session_state.get("dedupe_pending_list", False):
                    if st.button("重複を削除（同名を1件残す）", type="primary", key="dedupe_list_btn"):
                        st.session_state["dedupe_pending_list"] = True
                        st.rerun()
                else:
                    if st.button("実行する（元に戻せません）", type="primary", key="dedupe_list_exec"):
                        kept: List[Dict[str, Any]] = []
                        seen: set[str] = set()
                        for f in st.session_state.foods:
                            n = str(f.get(COL_NAME, "")).strip()
                            if not n or n in seen:
                                continue
                            seen.add(n)
                            kept.append(f)
                        st.session_state.foods = kept
                        save_foods(st.session_state.foods)
                        st.success("重複データを整理しました。")
                        st.session_state["dedupe_pending_list"] = False
                        st.rerun()
                    if st.button("キャンセル", key="dedupe_list_cancel"):
                        st.session_state["dedupe_pending_list"] = False
                        st.rerun()

            st.subheader("削除")
            names = [f.get(COL_NAME, "") for f in foods]
            unique_names = [n for n in names if n]

            if not unique_names:
                st.info("削除できる商品がありません。")
            else:
                remove_target = st.selectbox("削除する商品", options=sorted(set(unique_names)))
                if st.button("削除する", type="primary"):
                    before = len(foods)
                    foods = [f for f in foods if f.get(COL_NAME) != remove_target]
                    st.session_state.foods = foods
                    save_foods(st.session_state.foods)
                    after = len(foods)
                    st.success(f"削除しました（{before} -> {after}件）。")
                    st.rerun()

    with tab_compare:
        st.subheader("栄養素の比較（棒グラフ）")
        if df.empty:
            st.info("まだ登録がありません。`登録` タブで追加してください。")
        else:
            all_names = list(df[COL_NAME].values)
            default = all_names[: min(5, len(all_names))]
            selected_names = st.multiselect("比較する商品", options=sorted(set(all_names)), default=default)

            if not selected_names:
                st.warning("商品を選択してください。")
            else:
                df_sel = df[df[COL_NAME].isin(selected_names)].copy()
                df_sel = df_sel.sort_values(by=COL_NAME)

                dry_compare = st.checkbox("乾物換算で比較（ウェットのみ）", value=False)
                if dry_compare:
                    st.info(
                        "乾物換算は、乾物値(%) = 栄養素(%) ÷ (100 - 水分%) × 100 に基づき、ウェット（高水分）を除水基準に換算することで、ドライフードと公平に比較できます。"
                    )

                nutrient_options = [
                    COL_PROTEIN,
                    COL_FAT,
                    COL_CARBS,
                    COL_PHOSPHORUS_PCT,
                    COL_CALCIUM,
                    COL_MAGNESIUM,
                ]
                selected_nutrients = st.multiselect(
                    "比較する栄養素（複数選択可）",
                    options=nutrient_options,
                    default=[COL_PROTEIN, COL_FAT, COL_CARBS],
                )
                if not selected_nutrients:
                    st.warning("栄養素を選択してください。")
                else:
                    dry_col_map = {
                        COL_PROTEIN: COL_PROTEIN_DRY,
                        COL_FAT: COL_FAT_DRY,
                        COL_CARBS: COL_CARBS_DRY,
                        COL_PHOSPHORUS_PCT: COL_PHOSPHORUS_DRY,
                        COL_CALCIUM: COL_CALCIUM_DRY,
                        COL_MAGNESIUM: COL_MAGNESIUM_DRY,
                    }
                    plot_cols = selected_nutrients if not dry_compare else [dry_col_map.get(c, c) for c in selected_nutrients]
                    nutrient_df = df_sel.set_index(COL_NAME)[plot_cols]
                    st.bar_chart(nutrient_df, use_container_width=True)

                st.divider()
                st.subheader("コスパ（100gあたり価格）")
                comp_cols = [COL_NAME, COL_PRICE, COL_CONTENT, COL_PRICE_100G, COL_REVIEW]
                st.dataframe(df_sel[comp_cols].sort_values(by=COL_PRICE_100G), use_container_width=True, hide_index=True)

    with tab_rank:
        st.subheader("総合スコアランキング")
        if df.empty:
            st.info("まだ登録がありません。`登録` タブで追加してください。")
        else:
            scored = compute_ranking(
                df,
                w_review=w_review,
                w_protein=w_protein,
                w_cost_100g=w_cost_100g,
                w_fat=w_fat,
                w_calories=w_calories,
                w_phosphorus=w_phosphorus,
                w_carbs=w_carbs,
                w_animal_protein=w_animal_protein,
                w_additive_free=w_additive_free,
            )
            scored = scored.sort_values(by=COL_SCORE, ascending=False)
            scored_show = scored[
                [
                    COL_NAME,
                    COL_FOOD_TYPE,
                    COL_ORIGIN,
                    COL_SCORE,
                    COL_REVIEW,
                    COL_PROTEIN,
                    COL_FAT,
                    COL_CAL,
                    COL_PRICE_100G,
                    COL_PHOSPHORUS,
                    COL_CARBS,
                    COL_ANIMAL_PROTEIN_PCT,
                ]
            ].copy()
            scored_show[COL_SCORE] = scored_show[COL_SCORE].map(lambda x: round(float(x), 4))
            scored_show[COL_PRICE_100G] = scored_show[COL_PRICE_100G].map(lambda x: round(float(x), 2))
            scored_show[COL_PROTEIN] = scored_show[COL_PROTEIN].map(lambda x: round(float(x), 2))
            scored_show[COL_FAT] = scored_show[COL_FAT].map(lambda x: round(float(x), 2))
            scored_show[COL_CAL] = scored_show[COL_CAL].map(lambda x: round(float(x), 1))
            scored_show[COL_REVIEW] = scored_show[COL_REVIEW].map(lambda x: round(float(x), 2))
            scored_show[COL_PHOSPHORUS] = scored_show[COL_PHOSPHORUS].map(
                lambda x: round(float(x), 1) if pd.notna(x) else x
            )
            scored_show[COL_CARBS] = scored_show[COL_CARBS].map(
                lambda x: round(float(x), 1) if pd.notna(x) else x
            )
            scored_show[COL_ANIMAL_PROTEIN_PCT] = scored_show[COL_ANIMAL_PROTEIN_PCT].map(
                lambda x: round(float(x), 1) if pd.notna(x) else x
            )

            st.dataframe(scored_show, use_container_width=True, hide_index=True)

            with st.expander("総合スコアの考え方"):
                st.write(
                    "各指標を登録済み商品の範囲で 0-1 に正規化し、重み付き合計で総合スコアを算出します。"
                )
                st.write("高いほど良い: 口コミ評価, タンパク質, 動物性タンパク質(%), 添加物なし（3項目すべて空）")
                st.write(
                    "低いほど良い: 100gあたり価格, 脂質, カロリー（想定）, リン(mg/100g), 炭水化物(%)"
                )


if __name__ == "__main__":
    main()

