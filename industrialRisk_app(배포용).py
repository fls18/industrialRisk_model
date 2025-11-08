# -*- coding: utf-8 -*-
# industrialRisk_app.py — 4업종(운수/건설/제조/기타) 축소 + 분포캘리브레이션/앵커 보정 통합판
# - 입력: 지역 / 사업장 규모 / 업종(4종) / 근속기간(년)
# - 모델(.keras → .h5) 안전 로딩, 실패 시 룩업/priors로 동작
# - 모델 평탄 자동 감지(α 자동 하향) + 양방향 컨트라스트 확장 + 앵커 타깃 끌어당김
# - 개선 Top3 + 체크리스트 + 디버그 패널
# - 기존 세부 업종(농업/임업/어업/광업 등)은 내부적으로 "기타"로 흡수

import os, json, pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="산업 안전 위험도 예측 시뮬레이터", page_icon="🏭", layout="centered")
st.title("산업 안전 위험도 예측 시뮬레이터")

# ─────────────────────────────────────────
# 경로/상수
# ─────────────────────────────────────────
MODEL_KERAS = "industrial_risk_model.keras"
MODEL_H5    = "industrial_risk_model.h5"
SCALER_PATH = "scaler.pkl"
NUMCOL_PATH = "num_cols.json"
ONEHOT_PATH = "onehot_meta.json"
LOOKUP_PATH = "risk_lookup.csv"

REGION_OPTIONS = [
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "경기","강원","충북","충남","전북","전남","경북","경남","제주"
]
SCALE_OPTIONS = ["소규모","중규모","대규모"]

# 입력 UI: 업종 4개만 노출
INDUSTRY_OPTIONS = ["운수업", "건설업", "제조업", "기타"]

# 세부 업종 → 4업종으로 축소 매핑(입력 방어)
COLLAPSE_TO_4 = {
    "운수업":"운수업", "건설업":"건설업", "제조업":"제조업",
    # 나머지는 전부 "기타"
    "농업":"기타", "임업":"기타", "어업":"기타", "광업":"기타",
    "서비스업":"기타", "정보통신업":"기타", "도소매업":"기타",
    "전기/가스업":"기타", "부동산업":"기타", "교육서비스업":"기타",
    "보건업":"기타", "예술/스포츠":"기타", "기타":"기타",
}

# ─────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────
def _safe_read_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def bin_tenure_years(years: float) -> str:
    x = float(years)
    if x <= 1:   return "P_0_1"
    if x <= 3:   return "P_2_3"
    if x <= 5:   return "P_4_5"
    if x <= 10:  return "P_6_10"
    return "P_11p"

# ─────────────────────────────────────────
# 안전 로더 (.keras → .h5 → None)
# ─────────────────────────────────────────
def safe_load_model():
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None
    if os.path.exists(MODEL_KERAS):
        try:
            return load_model(MODEL_KERAS, compile=False)
        except Exception as e:
            st.warning(f"모델(.keras) 로딩 실패: {e}")
    if os.path.exists(MODEL_H5):
        try:
            return load_model(MODEL_H5, compile=False)
        except Exception as e:
            st.warning(f"모델(.h5) 로딩 실패: {e}")
    return None

def safe_load_meta():
    scaler=None; NUM=[]; CAT=[]
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH,"rb") as f: scaler=pickle.load(f)
        except Exception as e:
            st.warning(f"스케일러 로딩 실패: {e}")
    if os.path.exists(NUMCOL_PATH):
        try:
            NUM = json.load(open(NUMCOL_PATH,encoding="utf-8"))["num_cols"]
        except Exception as e:
            st.warning(f"num_cols 로딩 실패: {e}")
    if os.path.exists(ONEHOT_PATH):
        try:
            CAT = json.load(open(ONEHOT_PATH,encoding="utf-8"))["columns"]
        except Exception as e:
            st.warning(f"onehot_meta 로딩 실패: {e}")
    return scaler, NUM, CAT

model = safe_load_model()
scaler, NUM, CAT = safe_load_meta()
have_model = (model is not None) and (scaler is not None) and (CAT is not None)
df_lookup = _safe_read_csv(LOOKUP_PATH)

# ─────────────────────────────────────────
# 입력 UI
# ─────────────────────────────────────────
st.subheader("입력값")
c1, c2 = st.columns(2)
with c1:
    region   = st.selectbox("지역", REGION_OPTIONS, index=0)
    industry_raw = st.selectbox("업종", INDUSTRY_OPTIONS, index=2)  # 기본: 제조업
with c2:
    scale    = st.selectbox("사업장 규모", SCALE_OPTIONS, index=0)
    tenure   = st.number_input("근속기간(년)", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
tenure_bucket = bin_tenure_years(tenure)
# 방어: 혹시 외부에서 다른 업종 문자열이 들어와도 4분류로 축소
industry = COLLAPSE_TO_4.get(industry_raw, industry_raw if industry_raw in INDUSTRY_OPTIONS else "기타")

# 사이드바 즉석 튜닝
st.sidebar.header("튜닝(선택)")
SPREAD_LOW  = st.sidebar.slider("하위 컨트라스트(≤0.5 확장)", 1.2, 3.0, 2.2, 0.1)
SPREAD_HIGH = st.sidebar.slider("상위 컨트라스트(≥0.5 확장)", 1.2, 3.0, 2.0, 0.1)
ALPHA_MODEL = st.sidebar.slider("모델 가중 α (모델↔priors 블렌드)", 0.0, 1.0, 0.55, 0.05)
ANCHOR_STRENGTH = st.sidebar.slider("앵커 강도(정합 가중 β)", 0.0, 1.0, 0.85, 0.05)

# ─────────────────────────────────────────
# 예측 벡터
# ─────────────────────────────────────────
def make_num_vec():
    if not have_model or scaler is None or NUM is None or len(NUM)==0:
        return np.zeros((1,0))
    try:
        if hasattr(scaler,"mean_") and len(getattr(scaler,"mean_"))==len(NUM):
            base = np.array(scaler.mean_, dtype=float).reshape(1,-1)
        else:
            base = np.zeros((1,len(NUM)))
        return scaler.transform(base)
    except Exception:
        return np.zeros((1,len(NUM)))

def make_cat_row(region_val, industry_val, scale_val, tenure_bucket_val):
    if not have_model or CAT is None or len(CAT)==0:
        return pd.DataFrame(np.zeros((1,0)))
    Xcat = pd.DataFrame(np.zeros((1,len(CAT))), columns=CAT, dtype=float)
    for k in [f"지역_{region_val}", f"업종_{industry_val}", f"규모_{scale_val}", f"근속버킷_{tenure_bucket_val}"]:
        if k in Xcat.columns: Xcat.loc[0,k] = 1.0
    # 세부 업종 원핫이 남아있을 가능성에 대비: 업종_* 중 우리가 선택 안한 건 0으로 유지 (기본 동작)
    return Xcat

def predict_model(region_val, industry_val, scale_val, tenure_bucket_val):
    if not have_model: return None
    num_vec = make_num_vec()
    Xcat = make_cat_row(region_val, industry_val, scale_val, tenure_bucket_val)
    file_dummy_cols = [c for c in (CAT or []) if c.startswith("파일_")]
    try:
        if file_dummy_cols:
            preds=[]
            for fcol in file_dummy_cols:
                Xc=Xcat.copy()
                Xc.loc[:,file_dummy_cols]=0.0
                Xc.loc[0,fcol]=1.0
                X=np.hstack([num_vec, Xc.values])
                preds.append(float(model.predict(X,verbose=0)[0][0]))
            if preds: return float(np.mean(preds))
            Xc=Xcat.copy(); Xc.loc[:,file_dummy_cols]=0.0
            X=np.hstack([num_vec, Xc.values])
            return float(model.predict(X,verbose=0)[0][0])
        else:
            X=np.hstack([num_vec, Xcat.values])
            return float(model.predict(X,verbose=0)[0][0])
    except Exception as e:
        st.warning(f"모델 추론 실패: {e}")
        return None

# ─────────────────────────────────────────
# 룩업 백업
# ─────────────────────────────────────────
def _norm01(x: pd.Series):
    s=pd.to_numeric(x, errors="coerce")
    vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax-vmin)<1e-12:
        return None
    return (s - vmin) / (vmax - vmin)

def lookup_score(region_val, industry_val):
    df=df_lookup
    if df.empty: return None
    if "업종" in df.columns and "지역" in df.columns:
        sub=df[(df["업종"].astype(str)==industry_val)&(df["지역"].astype(str)==region_val)]
        if sub.empty: sub=df[df["업종"].astype(str)==industry_val]
    else:
        sub=df[df["업종"].astype(str)==industry_val] if "업종" in df.columns else df
    if sub.empty: sub=df
    if "Risk_Score_0_1" in sub.columns:
        arr=pd.to_numeric(sub["Risk_Score_0_1"], errors="coerce").dropna().values
        if arr.size and not np.allclose(arr.min(), arr.max()):
            return float(arr.mean())
    if "가중사건" in sub.columns:
        n=_norm01(sub["가중사건"])
        if n is not None: return float(pd.to_numeric(n, errors="coerce").mean())
    if "Risk_Score_0_1" in df.columns and df["Risk_Score_0_1"].notna().any():
        return float(pd.to_numeric(df["Risk_Score_0_1"], errors="coerce").mean())
    return 0.5

# ─────────────────────────────────────────
# 사실감 Priors (4업종 체계)
# ─────────────────────────────────────────
INDUSTRY_BASE = {
    "건설업": 0.62,
    "운수업": 0.56,
    "제조업": 0.52,
    "기타":   0.45,  # 농업/임업/어업/광업 등 포함
}
SCALE_ADD = {"소규모": -0.05, "중규모": +0.02, "대규모": +0.05}

def tenure_add(years: float, industry_val: str) -> float:
    # 4업종에 맞춘 직관적 가감: 초반(≤1년) 상승, 장기 하향
    if industry_val in ("기타",):
        # 기타(농업/임업/어업/광업 포함) 과도한 초반 상승 억제
        if years <= 1:  return -0.02
        if years <= 3:  return  0.00
        if years <= 5:  return -0.02
        if years <= 10: return -0.05
        return -0.08
    else:
        if years <= 1:  return +0.08
        if years <= 3:  return +0.03
        if years <= 5:  return -0.02
        if years <= 10: return -0.05
        return -0.08

def prior_score(industry_val, scale_val, years):
    base = INDUSTRY_BASE.get(industry_val, 0.45)
    sadd = SCALE_ADD.get(scale_val, 0.0)
    tadd = tenure_add(years, industry_val)
    return float(np.clip(base + sadd + tadd, 0.0, 1.0))

# ─────────────────────────────────────────
# 캘리브레이션: 블렌딩 → 컨트라스트 → 앵커
# ─────────────────────────────────────────
def estimate_model_flatness(ind, sca, ten_bucket):
    """파일_* 더미 평균으로 간이 분산 측정 → 너무 평탄하면 α 자동 낮춤"""
    if not have_model or CAT is None: return None
    num_vec = make_num_vec()
    # 지역 영향 제거 목적: 지역 더미는 0으로
    Xcat_base = make_cat_row("", ind, sca, ten_bucket)
    # 지역_* 컬럼 0으로 강제
    for c in [c for c in Xcat_base.columns if c.startswith("지역_")]:
        Xcat_base[c] = 0.0
    file_cols = [c for c in CAT if c.startswith("파일_")]
    if not file_cols: return None
    vals=[]
    for fc in file_cols:
        Xc=Xcat_base.copy()
        Xc.loc[:, file_cols]=0.0
        if fc in Xc.columns: Xc.loc[0, fc]=1.0
        X=np.hstack([num_vec, Xc.values])
        try:
            vals.append(float(model.predict(X, verbose=0)[0][0]))
        except:
            pass
    if len(vals)<2: return None
    return float(np.std(vals))

def blend_model_and_prior(p_model, p_prior, alpha):
    if p_model is None or not np.isfinite(p_model):
        return p_prior
    return float(np.clip(alpha*p_model + (1.0-alpha)*p_prior, 0.0, 1.0))

def contrast_stretch(p, k_low, k_high):
    if p >= 0.5:
        return float(np.clip(0.5 + (p-0.5)*k_high, 0.0, 1.0))
    else:
        return float(np.clip(0.5 - (0.5-p)*k_low, 0.0, 1.0))

# 앵커 타깃(4업종 체계) — 조합별 목표점으로 끌어당김
ANCHOR_TARGETS = {
    # 저위험 보장(제조업 소규모 장기/중장기)
    ("제조업", "소규모", "P_11p"): 0.22,
    ("제조업", "소규모", "P_6_10"): 0.25,

    # 중간 유지(운수 중규모 2~3년)
    ("운수업", "중규모", "P_2_3"): 0.53,

    # 고위험 보장(건설 초반 중/대규모)
    ("건설업", "대규모", "P_0_1"): 0.78,
    ("건설업", "중규모", "P_0_1"): 0.72,

    # 기타(농업/임업/어업/광업 등 흡수) — 너무 중간에 몰리지 않게 양끝 앵커 배치
    ("기타", "소규모", "P_11p"): 0.24,
    ("기타", "대규모", "P_0_1"): 0.58,
}

def get_anchor(industry_val, scale_val, ten_bucket):
    key = (industry_val, scale_val, ten_bucket)
    if key in ANCHOR_TARGETS:
        return ANCHOR_TARGETS[key], 1.0  # 완전 일치
    # 부분 일치(업종+규모) → 약한 앵커
    for k, v in ANCHOR_TARGETS.items():
        if k[0]==industry_val and k[1]==scale_val:
            return v, 0.6
    # 업종만 일치 → 더 약한 앵커
    for k, v in ANCHOR_TARGETS.items():
        if k[0]==industry_val:
            return v, 0.35
    return None, 0.0

def pull_to_anchor(p, industry_val, scale_val, ten_bucket, beta_user):
    t, match = get_anchor(industry_val, scale_val, ten_bucket)
    if t is None or match<=0.0:
        return p, 0.0, None
    beta = beta_user * match
    p2 = float((1.0-beta)*p + beta*t)
    return p2, beta, t

def hard_caps(p, industry_val, scale_val, years):
    # 4업종 정책 캡(필요 시 확장)
    if industry_val == "건설업" and years <= 1 and scale_val in ("중규모","대규모"):
        p = max(p, 0.60)  # 최소 주의 이상
    return float(np.clip(p, 0.0, 1.0))

def risk_band_msg(p):
    # 0~0.22 초록 / 0.22~0.50 노랑 / 0.50~0.70 주의 / 0.70~1.00 빨강
    if p > 0.70: return "🔴 고위험: 즉시 위험요인 점검 필요"
    if p > 0.50: return "🟠 주의: 관리감독자 순회 강화 및 TBM 집중"
    if p > 0.22: return "🟡 중위험: 신규·미숙련자 관리 필요"
    return "🟢 저위험: 현재 수준 유지 및 모니터링"

# ─────────────────────────────────────────
# 개선 제안 Top3 + 체크리스트
# ─────────────────────────────────────────
def top3_actions(p, scale_val, years):
    tips=[]
    if p > 0.70:
        tips += [
            "금일 작업 전 **TBM(위험성 확인) + 보호구 적합성 재점검**",
            "**밀폐·고소·협착** 구간 즉시 개선(가드, 추락방지, 비상정지)",
            "**사고다발 작업 순서 변경/속도 제한**(관리감독자 상시 순회)",
        ]
    elif p > 0.50:
        tips += [
            "**신규/전배치자 OJT** 1:1 동행 점검(체크리스트 서명)",
            "작업라인 **미끄럼·걸림·낙하물** 위험 개선(정리정돈/표지)",
            "**근골격계 부담작업 휴식주기** 도입(포지션 로테이션)",
        ]
    elif p > 0.22:
        tips += [
            "**표준작업 준수율** 주간 점검(표준서 최신화)",
            "소규모 개선: **표지판·통로 폭·조도** 보완",
            "월 1회 **사고사례 리마인드 교육**(5분 안전회의)",
        ]
    else:
        tips += [
            "기준 준수 유지: **정리정돈·표지·조도** 주기 점검",
            "분기 1회 **무재해 우수사례 공유**",
            "교육: **5분 안전회의**로 안전수칙 리마인드",
        ]
    if scale_val == "대규모":
        tips.insert(0, "라인/공정별 **관리감독자 순회 주기 단축** 및 즉시 시정조치")
    elif scale_val == "소규모":
        tips.insert(0, "**다기능 교육**로 인력 공백 시 안전역량 유지(겸직자 교육)")
    if years <= 1:
        tips.insert(0, "**1년 이하 작업자 고위험 공정 배치 제한 + 멘토 지정**")
    elif years <= 3:
        tips.insert(0, "**숙련도 상승기 오판 방지 교육**(안전 습관 고착)")
    out=[]
    for t in tips:
        if t not in out: out.append(t)
        if len(out)==3: break
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
    done=[]
    for i,item in enumerate(CHECKLIST_ITEMS):
        with cols[i%2]:
            done.append(st.checkbox(item, key=f"chk_{i}"))
    st.caption(f"완료: {sum(done)}/{len(done)}")

# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if st.button("위험도 계산하기", use_container_width=True):
    # 1) 모델/룩업
    pathway = "모델 예측"
    p_model = predict_model(region, industry, scale, tenure_bucket)
    if p_model is None or not np.isfinite(p_model):
        pathway = "룩업 평균"
        p_model = lookup_score(region, industry)
    if p_model is None or not np.isfinite(p_model):
        pathway = "기본값(0.5)"
        p_model = 0.5

    # 1-α 자동 조정: 모델 평탄하면 α 낮춤
    auto_alpha = float(ALPHA_MODEL)
    flat = estimate_model_flatness(industry, scale, tenure_bucket)
    if flat is not None and flat < 0.02:
        auto_alpha = min(auto_alpha, 0.30)

    # 2) priors
    p_prior = prior_score(industry, scale, float(tenure))

    # 3) 블렌딩 → 컨트라스트 → 앵커 → 하드캡
    p_blend   = blend_model_and_prior(float(p_model), p_prior, auto_alpha)
    p_stretch = contrast_stretch(p_blend, float(SPREAD_LOW), float(SPREAD_HIGH))
    p_anchor, beta_eff, target = pull_to_anchor(p_stretch, industry, scale, tenure_bucket, float(ANCHOR_STRENGTH))
    p_final  = hard_caps(p_anchor, industry, scale, float(tenure))

    # 4) 출력
    st.markdown(f"### 예측 위험도: **{p_final:.3f}**")
    st.write(risk_band_msg(p_final))

    st.markdown("#### 개선 제안 (Top 3)")
    for i, tip in enumerate(top3_actions(p_final, scale, float(tenure)), start=1):
        st.write(f"{i}. {tip}")

    checklist_ui()

    with st.expander("진단/디버그"):
        st.write(f"- 경로: **{pathway}**  | 모델평탄도 σ≈{0.0 if flat is None else flat:.3f} → α={auto_alpha:.2f}")
        st.write(f"- p_model: {p_model:.3f} / p_prior: {p_prior:.3f} / blend: {p_blend:.3f}")
        st.write(f"- stretch(k_low={SPREAD_LOW:.2f}, k_high={SPREAD_HIGH:.2f}) → {p_stretch:.3f}")
        if target is not None:
            st.write(f"- anchor(target={target:.2f}, β_eff={beta_eff:.2f}) → {p_anchor:.3f}")
        st.write(f"- hard caps → **{p_final:.3f}**")

# 사이드바 상태
st.sidebar.header("로딩 상태")
st.sidebar.write(f"모델: {'✅' if model is not None else '❌'}  (.keras→.h5 자동)")
st.sidebar.write(f"스케일러: {'✅' if scaler is not None else '❌'}")
st.sidebar.write(f"원핫 메타: {'✅' if CAT is not None else '❌'}  (열 수: {0 if CAT is None else len(CAT)})")
