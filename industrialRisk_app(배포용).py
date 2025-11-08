# -*- coding: utf-8 -*-
# industrialRisk_app.py — TF 지연 임포트(없어도 동작), priors/체크리스트 포함

import os, json, pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="산업 안전 위험도 예측 시뮬레이터", page_icon="🏭", layout="centered")
st.title("산업 안전 위험도 예측 시뮬레이터")

# 경로
MODEL_H5    = "industrial_risk_model.h5"
MODEL_KERAS = "industrial_risk_model.keras"
SCALER_PATH = "scaler.pkl"
NUMCOL_PATH = "num_cols.json"
ONEHOT_PATH = "onehot_meta.json"
LOOKUP_PATH = "risk_lookup.csv"

# 선택지(고정)
REGION_OPTIONS = ["서울","부산","대구","인천","광주","대전","울산","세종","경기","강원","충북","충남","전북","전남","경북","경남","제주"]
SCALE_OPTIONS  = ["소규모","중규모","대규모"]
INDUSTRY_OPTIONS = ["운수업","건설업","임업","제조업","광업","어업","농업","기타"]

def bin_tenure_years(years: float) -> str:
    x = float(years)
    if x <= 1:   return "P_0_1"
    if x <= 3:   return "P_2_3"
    if x <= 5:   return "P_4_5"
    if x <= 10:  return "P_6_10"
    return "P_11p"

def _safe_read_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

# --------- 안전한 모델 로더 (지연 임포트) ----------
def safe_load_model():
    try:
        from tensorflow.keras.models import load_model  # 지연 임포트 (없으면 except로 감)
    except Exception:
        return None
    # .keras 우선
    if os.path.exists(MODEL_KERAS):
        try:
            return load_model(MODEL_KERAS, compile=False)
        except Exception as e:
            st.warning(f"모델(.keras) 로딩 실패: {e}")
    # .h5 백업
    if os.path.exists(MODEL_H5):
        try:
            return load_model(MODEL_H5, compile=False)
        except Exception as e:
            st.warning(f"모델(.h5) 로딩 실패: {e}")
    return None

def safe_load_meta():
    scaler = None; NUM=[]; CAT=[]
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH,"rb") as f: scaler = pickle.load(f)
        except Exception as e: st.warning(f"스케일러 로딩 실패: {e}")
    if os.path.exists(NUMCOL_PATH):
        try:
            NUM = json.load(open(NUMCOL_PATH,encoding="utf-8"))["num_cols"]
        except Exception as e: st.warning(f"num_cols 로딩 실패: {e}")
    if os.path.exists(ONEHOT_PATH):
        try:
            CAT = json.load(open(ONEHOT_PATH,encoding="utf-8"))["columns"]
        except Exception as e: st.warning(f"onehot_meta 로딩 실패: {e}")
    return scaler, NUM, CAT

model = safe_load_model()
scaler, NUM, CAT = safe_load_meta()
have_model = (model is not None) and (scaler is not None) and (CAT is not None)

df_lookup = _safe_read_csv(LOOKUP_PATH)

# --------- 입력 UI ----------
st.subheader("입력값")
c1, c2 = st.columns(2)
with c1:
    region = st.selectbox("지역", REGION_OPTIONS, index=0)
    industry = st.selectbox("업종", INDUSTRY_OPTIONS, index=3)
with c2:
    scale = st.selectbox("사업장 규모", SCALE_OPTIONS, index=0)
    tenure = st.number_input("근속기간(년)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

tenure_bucket = bin_tenure_years(tenure)

# --------- 예측 벡터 ----------
def make_num_vec():
    if not have_model or scaler is None or NUM is None or len(NUM)==0:
        return np.zeros((1, 0))
    if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_")) == len(NUM):
        base = np.array(scaler.mean_, dtype=float).reshape(1, -1)
    else:
        base = np.zeros((1, len(NUM)), dtype=float)
    try:
        return scaler.transform(base)
    except Exception:
        return np.zeros((1, len(NUM)))

def make_cat_row(region_val, industry_val, scale_val, tenure_bucket_val):
    if not have_model or CAT is None or len(CAT)==0:
        return pd.DataFrame(np.zeros((1,0)))
    Xcat = pd.DataFrame(np.zeros((1, len(CAT))), columns=CAT, dtype=float)
    for k in [f"지역_{region_val}", f"업종_{industry_val}", f"규모_{scale_val}", f"근속버킷_{tenure_bucket_val}"]:
        if k in Xcat.columns: Xcat.loc[0, k] = 1.0
    return Xcat

def predict_model(region_val, industry_val, scale_val, tenure_bucket_val):
    if not have_model:
        return None
    num_vec = make_num_vec()
    Xcat_base = make_cat_row(region_val, industry_val, scale_val, tenure_bucket_val)
    file_dummy_cols = [c for c in (CAT or []) if c.startswith("파일_")]
    try:
        if file_dummy_cols:
            preds=[]
            for fcol in file_dummy_cols:
                Xc = Xcat_base.copy()
                Xc.loc[:, file_dummy_cols] = 0.0
                Xc.loc[0, fcol] = 1.0
                X = np.hstack([num_vec, Xc.values])
                p = float(model.predict(X, verbose=0)[0][0])
                preds.append(p)
            if preds: return float(np.mean(preds))
        X = np.hstack([num_vec, Xcat_base.values])
        return float(model.predict(X, verbose=0)[0][0])
    except Exception as e:
        st.warning(f"모델 추론 실패: {e}")
        return None

# --------- 룩업 백업 ----------
def _norm01(x: pd.Series):
    s = pd.to_numeric(x, errors="coerce")
    vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12: return None
    return (s - vmin) / (vmax - vmin)

def lookup_score(region_val, industry_val):
    if df_lookup.empty: return None
    df = df_lookup.copy()
    if "업종" in df.columns and "지역" in df.columns:
        sub = df[(df["업종"].astype(str)==industry_val) & (df["지역"].astype(str)==region_val)]
        if sub.empty: sub = df[df["업종"].astype(str)==industry_val]
    else:
        sub = df[df["업종"].astype(str)==industry_val] if "업종" in df.columns else df
    if sub.empty: sub = df
    if "Risk_Score_0_1" in sub.columns:
        arr = pd.to_numeric(sub["Risk_Score_0_1"], errors="coerce").dropna().values
        if arr.size and not np.allclose(arr.min(), arr.max()): return float(arr.mean())
    if "가중사건" in sub.columns:
        n = _norm01(sub["가중사건"])
        if n is not None: return float(pd.to_numeric(n, errors="coerce").mean())
    if "Risk_Score_0_1" in df.columns and df["Risk_Score_0_1"].notna().any():
        return float(pd.to_numeric(df["Risk_Score_0_1"], errors="coerce").mean())
    return 0.5

# --------- Priors & 출력 보정 ----------
INDUSTRY_UPLIFT = {"건설업":0.12, "광업":0.10, "제조업":0.05}
SCALE_UPLIFT    = {"대규모":0.05, "중규모":0.02, "소규모":0.00}

def tenure_uplift(years: float) -> float:
    if years <= 1: return 0.15
    if years <= 3: return 0.08
    return 0.00

def apply_priors(p, industry_val, scale_val, tenure_years):
    u = 0.0
    u += INDUSTRY_UPLIFT.get(industry_val, 0.0)
    u += SCALE_UPLIFT.get(scale_val, 0.0)
    u += tenure_uplift(tenure_years)
    return float(np.clip(p + u, 0.0, 1.0))

def pretty_scale(p):
    return float(np.clip(0.5 + (p - 0.5) * 1.1, 0.0, 1.0))

def risk_band_msg(p):
    if p > 0.65: return "🔴 고위험: 즉시 위험요인 점검 필요"
    if p > 0.35: return "🟠 중위험: 작업 전 점검·OJT 강화 권장"
    return "🟢 저위험: 현재 수준 유지 및 모니터링"

def top3_actions(p, scale_val, tenure_years):
    tips = []
    if p > 0.65:
        tips += [
            "금일 작업 전 **TBM(위험성 확인) + 보호구 적합성 재점검**",
            "**밀폐·고소·협착** 구간 즉시 개선(가드, 추락방지, 비상정지)",
            "**사고다발 작업 순서 변경/속도 제한**(관리감독자 상시 순회)",
        ]
    elif p > 0.35:
        tips += [
            "**신규/전배치자 OJT** 1:1 동행 점검(체크리스트 서명)",
            "작업라인 **미끄럼·걸림·낙하물** 위험 개선(정리정돈/표지)",
            "**근골격계 부담작업 휴식주기** 도입(포지션 로테이션)",
        ]
    else:
        tips += [
            "**표준작업 준수율** 주간 점검(표준서 최신화)",
            "소규모 개선: **표지판·통로 폭·조도** 보완",
            "월 1회 **사고사례 리마인드 교육**(5분 안전회의)",
        ]
    if scale_val == "대규모":
        tips.insert(0, "라인/공정별 **관리감독자 순회 주기 단축** 및 즉시 시정조치")
    elif scale_val == "소규모":
        tips.insert(0, "**다기능 교육**로 인력 공백 시 안전역량 유지(겸직자 교육)")
    if tenure_years <= 1:
        tips.insert(0, "**1년 이하 작업자 고위험 공정 배치 제한 + 멘토 지정**")
    elif tenure_years <= 3:
        tips.insert(0, "**숙련도 상승기 오판 방지 교육**(안전 습관 고착)")
    out = []
    for t in tips:
        if t not in out: out.append(t)
        if len(out) == 3: break
    return out

CHECKLIST_ITEMS = [
    "작업 전 TBM(위험성 확인) 실시 및 서명",
    "추락·끼임 방지 가드/난간/덮개 이상 없음",
    "비상정지 스위치/차단장치 작동 확인",
    "개인보호구(PPE) 착용·적합성 점검 완료",
    "통로·바닥 정리정돈(미끄럼·걸림 요인 제거)",
    "전원/에너지 차단(LOTO) 절차 준수",
    "중량물 취급 보조장비 사용 및 무리 작업 금지",
    "신규/전배치 근로자 OJT 기록 및 멘토 배정",
]

def checklist_ui():
    st.markdown("#### 현장 체크리스트")
    cols = st.columns(2)
    done = []
    for i, item in enumerate(CHECKLIST_ITEMS):
        with cols[i % 2]:
            done.append(st.checkbox(item, key=f"chk_{i}"))
    st.caption(f"완료: {sum(done)}/{len(done)}")

# 실행
if st.button("위험도 계산하기", use_container_width=True):
    pathway = "모델 예측"
    pred = None
    try:
        pred = predict_model(region, industry, scale, tenure_bucket)
    except Exception as e:
        st.warning(f"모델 예측 경고: {e}")
        pred = None

    if pred is None or not np.isfinite(pred):
        pathway = "룩업 평균(보정)"
        pred = lookup_score(region, industry)

    if pred is None or not np.isfinite(pred):
        pathway = "기본값(0.5)"
        pred = 0.5

    p = pretty_scale(apply_priors(float(pred), industry, scale, float(tenure)))

    st.markdown(f"### 예측 위험도: **{p:.3f}**")
    st.write(risk_band_msg(p))

    st.markdown("#### 개선 제안 (Top 3)")
    for i, tip in enumerate(top3_actions(p, scale, float(tenure)), start=1):
        st.write(f"{i}. {tip}")

    checklist_ui()

# 상태표시
st.sidebar.header("로딩 상태")
st.sidebar.write(f"모델: {'✅' if have_model else '❌'} (없으면 룩업 모드)")
