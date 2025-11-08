# -*- coding: utf-8 -*-
"""
산업 안전 위험도 예측 - '기타' 쏠림 강제 보정판 (wide 비는 문제 해결)
- 25개 엑셀 자동 스캔
- 헤더 추정/두줄헤더/중복컬럼/합계행 제거/숫자화 안전 처리
- 업종/지역 자동 승격(지역명 사전)
- ★ 측정항목이 모두 '기타'일 때 파일 종류로 분류 강제(근속기간→근속버킷 매핑)
- long → wide → 임시 Risk_Score 생성
- n≥20일 때만 Keras 학습(아니면 lookup 저장)
"""

import os, re, glob, json, pickle, warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

warnings.filterwarnings("ignore", category=FutureWarning)

# ── (선택) 수동 오버라이드 ──
OVERRIDES: Dict[str, Dict] = {
    # "재해정도_2019": {"header_row": 1, "sheet_name": 0, "ids": ["업종","재해정도"]},
}

PATTERNS = ["*규모*.xlsx","*근속기간*.xlsx","*발생형태*.xlsx","*재해정도*.xlsx","*지역*.xlsx"]
BAN_NAMES = {"industrial_risk_model.h5","scaler.pkl","industrial_clean.csv","industrial_risk.csv",
             "num_cols.json","onehot_meta.json","group_means.json","training_curve.png",
             "risk_lookup.csv","feature_lookup.csv"}

REGION_NAMES = {
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "경기","강원","충북","충남","전북","전남","경북","경남","제주",
    "서울특별시","부산광역시","대구광역시","인천광역시","광주광역시","대전광역시","울산광역시",
    "경기도","강원도","충청북도","충청남도","전라북도","전라남도","경상북도","경상남도","제주특별자치도",
}

def list_all_files(patterns: List[str]) -> List[str]:
    files=[]
    for p in patterns: files += glob.glob(p)
    return sorted([f for f in files if os.path.basename(f) not in BAN_NAMES])

def extract_year_from_name(path: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", os.path.basename(path))
    return int(m.group(1)) if m else None

def detect_kind_from_name(path: str) -> str:
    n = os.path.basename(path)
    if "규모" in n: return "규모"
    if "근속기간" in n: return "근속기간"
    if "발생형태" in n: return "발생형태"
    if "재해정도" in n: return "재해정도"
    if "지역" in n: return "지역"
    return "기타"

def _stripspaces(x: str) -> str:
    return re.sub(r"\s+", "", str(x).replace("\u00a0"," ").strip())

def normalize_cols(cols: list) -> list:
    out=[]
    for c in cols:
        cc = _stripspaces(c)
        cc = (cc.replace("대업종","업종")
                .replace("연 도","연도").replace("연도도","연도")
                .replace("근속(년)","근속기간")
                .replace("깔림/뒤집힘","깔림.뒤집힘").replace("깔림.뒤집힘","깔림.뒤집힘"))
        if cc in ("","None","nan"): cc = None
        out.append(cc)
    fixed=[]
    for i,h in enumerate(out):
        fixed.append(f"col_{i}" if (not h or (isinstance(h,str) and h.startswith("Unnamed"))) else h)
    return fixed

def _convert_series_to_numeric(s: pd.Series) -> pd.Series:
    return (s.astype(str)
             .str.replace(",","",regex=False)
             .str.replace("%","",regex=False)
             .str.replace("\u00a0","",regex=False)
             .str.replace("−","-",regex=False)
             .str.strip()
             .replace({"":"0","-":"0","nan":"0","None":"0"})
             .pipe(pd.to_numeric, errors="coerce")
             .fillna(0))

def safe_to_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns: 
            continue
        target = df.loc[:, c]  # 중복명일 수 있음
        if isinstance(target, pd.DataFrame):
            for sub in target.columns:
                df[sub] = _convert_series_to_numeric(df[sub])
        else:
            df[c] = _convert_series_to_numeric(target)
    return df

KNOWN_EVENT_COLS = {
    "떨어짐","넘어짐","부딪힘","물체에맞음","무너짐","끼임","절단베임찔림","감전",
    "폭발파열","화재","깔림.뒤집힘","이상온도물체접촉","빠짐익사","불균형및무리한동작",
    "사업장외교통사고","업무상질병","체육행사","폭력행위","동물상해","기타","분류불능",
    "깔림","뒤집힘"
}
def _looks_scale_bucket(name: str) -> bool:
    x = str(name)
    return bool(re.search(r"(\d+\s*~\s*\d+인|\d+\s*-\s*\d+인|\d+인미만|\d+인이상|\d+인|소규모|중규모|대규모)", x))

def _looks_tenure_bucket(name: str) -> bool:
    x = str(name)
    return bool(re.search(r"(P_0_1|P_2_3|P_4_5|P_6_10|P_11p|\d+년미만|\d+\s*~\s*\d+년|\d+년이상)", x))

def classify_dim(colname: str) -> str:
    x = str(colname)
    if x in KNOWN_EVENT_COLS: return "발생형태"
    if _looks_scale_bucket(x): return "규모"
    if _looks_tenure_bucket(x): return "근속버킷"
    if re.search(r"(사망|중상|경상|휴업|재해정도)", x): return "재해정도"
    return "기타"

def apply_override(path: str):
    b = os.path.basename(path)
    for k,cfg in OVERRIDES.items():
        if k in b: return cfg
    return {}

def read_excel_any(path: str, sheet_name=None) -> pd.DataFrame:
    target = 0 if sheet_name is None else sheet_name
    try:
        df = pd.read_excel(path, sheet_name=target, header=None)
    except Exception:
        df = pd.read_excel(path, sheet_name=target, header=None, engine="xlrd")
    if isinstance(df, dict):
        for _,sdf in df.items():
            if isinstance(sdf,pd.DataFrame) and not sdf.dropna(how="all").empty and sdf.shape[1]>0:
                return sdf
        return next(iter(df.values()))
    return df

def guess_header_row(df: pd.DataFrame) -> int:
    tokens = {"업종","지역","규모","근속기간","발생형태","재해정도","연도","합계","계","총계"}
    best_row, best_score = 0, -1e9
    for r in range(min(6,len(df))):
        vals = df.iloc[r].astype(str).tolist()
        n_nonempty = sum(v.strip() not in ("","nan","None") for v in vals)
        if n_nonempty==0: continue
        score=0
        for v in vals:
            v2 = _stripspaces(v)
            if v2 in tokens: score+=4
            if v2 in KNOWN_EVENT_COLS: score+=2
            if v2.startswith("Unnamed"): score-=1
        score += n_nonempty*0.1
        if score>best_score: best_score, best_row = score, r
    return best_row

def repair_duplicate_headers(body: pd.DataFrame) -> pd.DataFrame:
    cols = list(body.columns)
    dup = body.columns.duplicated(keep=False).any() or any(re.match(r".+\.\d+$", c) for c in cols)
    if dup and any(c.startswith("업종") for c in cols):
        first = body.iloc[0].astype(str).str.replace("\u00a0","",regex=False).str.strip()
        new_cols=[]
        for c in cols:
            if c.startswith("업종") and str(first.get(c,"")) not in ("","nan","None"):
                new_cols.append(_stripspaces(first[c]))
            else:
                new_cols.append(c)
        body = body.copy()
        body.columns = normalize_cols(new_cols)
        body = body.iloc[1:].reset_index(drop=True)
    return body

def _try_promote_nextrow_as_header(body: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if body.empty: return body, False
    first_row = body.iloc[0]
    num_ratio = pd.to_numeric(first_row, errors="coerce").notna().mean()
    if num_ratio >= 0.7:
        new_cols = normalize_cols(first_row.tolist())
        body2 = body.iloc[1:].reset_index(drop=True).copy()
        if len(new_cols) >= body2.shape[1]:
            body2.columns = new_cols[:body2.shape[1]]
            return body2, True
    return body, False

def infer_header_and_fix(df: pd.DataFrame, force_header_row: Optional[int]=None) -> Tuple[pd.DataFrame,list,int]:
    hr = guess_header_row(df) if force_header_row is None else int(force_header_row)
    header = df.iloc[hr].astype(str).tolist()
    body = df.iloc[hr+1:].reset_index(drop=True).copy()
    body = body.dropna(axis=1, how="all")
    cols = normalize_cols(header[:len(body.columns)])
    body.columns = cols
    body = body.dropna(how="all").reset_index(drop=True)
    body = repair_duplicate_headers(body)
    body, _ = _try_promote_nextrow_as_header(body)
    return body, list(body.columns), hr

def _series_of(df: pd.DataFrame, colname: str) -> pd.Series:
    obj = df.loc[:, colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

def _uniq_value_cols(body: pd.DataFrame, value_cols: list, ids: list) -> list:
    seen = {}
    new_cols = []
    for c in value_cols:
        if c in ids: 
            continue
        name = c
        while (body.columns == name).sum() > 1 or name in seen:
            k = seen.get(c, 0) + 1
            name = f"{c}__dup{k}"
            seen[c] = k
        if name != c:
            idx = list(body.columns).index(c)
            cols = list(body.columns); cols[idx] = name
            body.columns = cols
        new_cols.append(name)
    return new_cols

def _maybe_promote_id_columns(body: pd.DataFrame, ids: list) -> list:
    if "업종" in ids and "지역" in ids: 
        return ids[:]
    candidates = []
    for c in body.columns:
        if c in ids: 
            continue
        s = _series_of(body, c)
        if s.isna().all(): 
            continue
        if pd.to_numeric(s, errors="coerce").notna().mean() >= 0.85:
            continue
        nunq = s.astype(str).str.strip().replace({"": "NaN"}).nunique(dropna=True)
        if 2 <= nunq <= 200:
            candidates.append(c)
    region_best = None; region_hit = 0
    for c in candidates:
        s = _series_of(body, c)
        vals = set(s.astype(str).str.strip())
        hit = len(vals & REGION_NAMES)
        if hit > region_hit:
            region_hit, region_best = hit, c
    new_ids = ids[:]
    if region_best and "지역" not in new_ids:
        body.rename(columns={region_best:"지역"}, inplace=True, errors="ignore")
        new_ids.append("지역")
    if "업종" not in new_ids:
        for c in candidates:
            if c == region_best: 
                continue
            body.rename(columns={c:"업종"}, inplace=True, errors="ignore")
            new_ids.append("업종")
            break
    return list(dict.fromkeys(new_ids))

# ── 파일→long ──
def melt_one_file(path: str) -> pd.DataFrame:
    cfg = apply_override(path)
    raw = read_excel_any(path, sheet_name=cfg.get("sheet_name", None))
    if "header_row" in cfg:
        body, cols, used_header_row = infer_header_and_fix(raw, force_header_row=int(cfg["header_row"]))
    else:
        body, cols, used_header_row = infer_header_and_fix(raw, force_header_row=None)

    body["파일"] = os.path.basename(path)
    kind = detect_kind_from_name(path)

    id_candidates = [c for c in ["업종","지역","규모","근속기간","발생형태","재해정도","연도","파일"] if c in body.columns]
    prefer_ids = {
        "규모":["업종","파일"],
        "근속기간":["업종","파일"],
        "발생형태":["업종","파일"],
        "재해정도":["업종","파일"],
        "지역":["업종","지역","파일"],
    }.get(kind, ["파일"])

    if "ids" in cfg:
        ids = [c for c in cfg["ids"] if c in body.columns]
    else:
        ids = [c for c in prefer_ids if c in body.columns]
        if not ids:
            ids = id_candidates[:2] if len(id_candidates)>=2 else id_candidates

    # 합계/계/총계 행 제거
    for k in ["업종","지역","규모","근속기간","발생형태","재해정도"]:
        if k in body.columns:
            s = _series_of(body, k)
            body[k] = s.astype(str).str.replace("\u00a0","",regex=False).str.strip()
            body = body[~body[k].astype(str).str.contains(r"(합계|총계|^계$)", na=False)]

    # 값 컬럼
    value_cols = [c for c in body.columns if c not in ids]
    drop_sum_cols = [c for c in value_cols if re.search(r"(합계|총계|^계$)", str(c))]
    if drop_sum_cols:
        body = body.drop(columns=drop_sum_cols, errors="ignore")
    value_cols = [c for c in value_cols if c not in drop_sum_cols]

    # 업종/지역 자동 승격
    ids = _maybe_promote_id_columns(body, ids)
    value_cols = [c for c in body.columns if c not in ids]

    # 값 컬럼 중복명 유니크화 + 숫자화
    value_cols = _uniq_value_cols(body, value_cols, ids)
    body = safe_to_numeric(body, value_cols)

    long_df = body.melt(id_vars=[c for c in ids if c in body.columns],
                        value_vars=value_cols,
                        var_name="측정항목", value_name="값")
    long_df = long_df[~long_df["측정항목"].astype(str).str.contains(r"(합계|총계|^계$)", na=False)]

    # 분류
    long_df["측정항목"] = long_df["측정항목"].astype(str).map(_stripspaces)
    long_df["분류"] = long_df["측정항목"].apply(classify_dim)

    # ★ 전부 '기타'인 경우: 파일 종류로 강제 보정
    if long_df["분류"].eq("기타").all():
        forced = "근속버킷" if kind=="근속기간" else kind
        if forced in {"규모","근속버킷","발생형태","재해정도"}:
            long_df["분류"] = forced

    long_df["연도"] = extract_year_from_name(path) or 0
    long_df["파일"] = os.path.basename(path)
    long_df["종류"] = kind
    long_df["__헤더행"] = used_header_row

    for c in ids + ["측정항목","분류"]:
        if c in long_df.columns:
            long_df[c] = long_df[c].astype(str).str.replace("\u00a0","",regex=False).str.strip()
    return long_df

# ── 로드 → long ──
FILES = list_all_files(PATTERNS)
if not FILES:
    raise SystemExit("❌ .xlsx 파일을 찾지 못했습니다. (파일명에 규모/근속기간/발생형태/재해정도/지역 포함)")

print("📂 스캔 대상:")
for f in FILES: print(" -", f)

all_long = pd.concat([melt_one_file(p) for p in FILES], ignore_index=True)
for c in ["업종","지역","규모","근속기간","발생형태","재해정도","측정항목","분류","파일"]:
    if c in all_long.columns:
        all_long[c] = all_long[c].astype(str).str.strip().replace({"nan":"", "None":""})

print("📏 long shape:", all_long.shape)
print("🔎 예시:\n", all_long.head(5))
print("📊 분류 카운트:\n", all_long["분류"].value_counts(dropna=False).to_dict())

# ── long → wide & 임시 라벨 ──
SEV_W = {"사망":1.0,"사망자":1.0,"중상":0.7,"중상자":0.7,"경상":0.4,"경상자":0.4,"휴업":0.5,"휴업재해":0.5}

def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for kind in ["발생형태","규모","근속버킷","재해정도"]:
        sub = df[df["분류"]==kind].copy()
        if sub.empty: 
            continue
        keys = [k for k in ["업종","지역","연도","파일"] if k in sub.columns]
        pvt = sub.pivot_table(index=keys, columns="측정항목", values="값", aggfunc="sum", fill_value=0)
        pvt.columns = [f"{kind}:{_stripspaces(c)}" for c in pvt.columns]
        out.append(pvt)
    if not out: 
        return pd.DataFrame()
    wide = pd.concat(out, axis=1).reset_index()
    wide.columns = [_stripspaces(c) for c in wide.columns]
    if "연도" not in wide.columns and "파일" in wide.columns:
        wide["연도"] = wide["파일"].apply(lambda x: extract_year_from_name(x) or 0)
    return wide

wide = build_wide(all_long)
if wide.empty:
    raise SystemExit("❌ 변환된 wide 테이블이 비었습니다. (강제 분류 후에도 비정상) — OVERRIDES에서 header_row/ids를 지정해보세요.")

scale_cols = [c for c in wide.columns if c.startswith("규모:")]
tenur_cols = [c for c in wide.columns if c.startswith("근속버킷:")]
event_cols = [c for c in wide.columns if c.startswith("발생형태:")]
sev_cols   = [c for c in wide.columns if c.startswith("재해정도:")]

wide["사건총합"] = 0.0
if event_cols:   wide["사건총합"] = wide[event_cols].sum(axis=1)
elif scale_cols: wide["사건총합"] = wide[scale_cols].sum(axis=1)
elif tenur_cols: wide["사건총합"] = wide[tenur_cols].sum(axis=1)

if sev_cols:
    def sev_sum(row):
        s=0.0
        for c in sev_cols:
            name = c.split(":",1)[1]
            w = SEV_W.get(name, 0.5)
            s += row[c]*w
        return s
    wide["가중사건"] = wide.apply(sev_sum, axis=1)
else:
    wide["가중사건"] = wide["사건총합"]

# 그룹키 우선순위
if all(k in wide.columns for k in ["업종","지역"]):
    group_keys = ["업종","지역"]
elif "업종" in wide.columns:
    group_keys = ["업종","파일"]
else:
    group_keys = ["파일"]
groupby_keys = list(dict.fromkeys(group_keys + (["연도"] if "연도" in wide.columns else [])))
print("\n🧩 그룹 키(표시용):", group_keys, " | 실제 groupby 키:", groupby_keys)

grp = wide.groupby(groupby_keys, as_index=False, sort=False)["가중사건"].mean()
y_min, y_max = float(grp["가중사건"].min()), float(grp["가중사건"].max())
grp["Risk_Score_0_1"] = 0.0 if (y_max-y_min)<1e-12 else (grp["가중사건"]-y_min)/(y_max-y_min)

def agg_share(df, cols, keys):
    if not cols or not keys: return pd.DataFrame()
    sub = df[keys + cols].copy()
    sub["sum"] = sub[cols].sum(axis=1)
    for c in cols:
        sub[c] = sub[c] / sub["sum"].replace(0, np.nan)
    sub = sub.drop(columns=["sum"])
    return sub.groupby(keys, as_index=False).mean(numeric_only=True)

keys_share = [k for k in ["업종","지역"] if k in wide.columns]
scale_share = agg_share(wide, scale_cols, keys_share) if keys_share else pd.DataFrame()
tenur_share = agg_share(wide, tenur_cols, keys_share) if keys_share else pd.DataFrame()

# 저장물
risk_lookup_cols = [c for c in ["업종","지역","파일","연도","가중사건","Risk_Score_0_1"] if c in grp.columns]
grp[risk_lookup_cols].to_csv("risk_lookup.csv", index=False, encoding="utf-8-sig")

feat_keys = [k for k in ["업종","지역"] if k in grp.columns] or [k for k in ["업종","파일"] if k in grp.columns] or [k for k in ["파일"] if k in grp.columns]
feat = grp[[c for c in feat_keys + ["연도","Risk_Score_0_1"] if c in grp.columns]].copy()
if not scale_share.empty and keys_share:
    feat = feat.merge(scale_share, on=keys_share, how="left")
if not tenur_share.empty and keys_share:
    feat = feat.merge(tenur_share, on=keys_share, how="left")
feat = feat.fillna(0)

print("🧩 feat shape:", feat.shape)
for k in [c for c in feat.columns if c in ["업종","지역","연도","파일"]]:
    try:
        print(f"   - {k} 고유값 수:", feat[k].nunique())
    except Exception:
        pass

# ── 학습 (충분할 때만) ──
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

n_samples = len(feat)
trained = False

if n_samples >= 20:
    cat_cols = [c for c in ["업종","지역","파일"] if c in feat.columns]
    num_cols = [c for c in feat.columns if c not in (cat_cols + ["Risk_Score_0_1"])]
    df_cat = feat[cat_cols].copy() if cat_cols else pd.DataFrame(index=feat.index)
    X_cat = pd.get_dummies(df_cat) if not df_cat.empty else pd.DataFrame(np.zeros((n_samples,0)))
    X_num = feat[num_cols].copy()
    scaler = StandardScaler()
    if X_num.shape[1]==0:
        X_num = pd.DataFrame(np.zeros((n_samples,1)), columns=["num_dummy"])
    num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=feat.index)

    X = np.hstack([num_scaled.values, X_cat.values])
    y = feat["Risk_Score_0_1"].astype(float).values

    test_size = 0.2 if n_samples >= 50 else max(1, n_samples//10)/n_samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=200, batch_size=16, verbose=1, callbacks=[es])
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ 테스트 MAE: {mae:.4f}")

    model.save("industrial_risk_model.h5"); print("🧠 저장: industrial_risk_model.h5")
    with open("scaler.pkl","wb") as f: pickle.dump(scaler,f)
    with open("num_cols.json","w",encoding="utf-8") as f: json.dump({"num_cols": list(num_scaled.columns)}, f, ensure_ascii=False, indent=2)
    with open("onehot_meta.json","w",encoding="utf-8") as f: json.dump({"columns": list(X_cat.columns)}, f, ensure_ascii=False, indent=2)

    feat_out = pd.concat([feat.reset_index(drop=True), pd.DataFrame(X_cat, columns=X_cat.columns)], axis=1)
    feat_out.to_csv("industrial_clean.csv", index=False, encoding="utf-8-sig"); print("📄 저장: industrial_clean.csv")
    trained = True
else:
    print(f"⚠️ 학습 샘플이 {n_samples}개라서 ML 학습 스킵. (룰-기반 lookup만 사용)")

if not os.path.exists("industrial_clean.csv"):
    feat.to_csv("industrial_clean.csv", index=False, encoding="utf-8-sig")

print("\n🎉 완료 산출물:")
if trained:
    print(" - industrial_risk_model.h5")
    print(" - scaler.pkl")
    print(" - num_cols.json")
    print(" - onehot_meta.json")
print(" - industrial_clean.csv")
print(" - risk_lookup.csv")
if os.path.exists("feature_lookup.csv"):
    print(" - feature_lookup.csv")
