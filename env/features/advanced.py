import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint, adfuller
except Exception:
    coint = None
    adfuller = None

try:
    from pykalman import KalmanFilter
except Exception:
    KalmanFilter = None


# ============================================================
# FEATURE COUNT (≈146 FINAL)
# ============================================================
N_ADVANCED_PAIR_FEATURES = 151
_EPS = 1e-9


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _pct_change(s):
    return s.pct_change().replace([np.inf, -np.inf], 0).fillna(0)

def _rolling_z(s, w):
    if len(s) < w: return 0.0
    s0 = s.iloc[-w:]
    return float((s0.iloc[-1] - s0.mean()) / (s0.std() + _EPS))

def _rolling_vol(s, w):
    if len(s) < w: return 0.0
    return float(s.iloc[-w:].std())

def _rolling_corr(a, b, w):
    if len(a) < w or len(b) < w: return 0.0
    return float(a.iloc[-w:].corr(b.iloc[-w:]))

def _poly_slope(s, w):
    if len(s) < w: return 0.0
    y = s.iloc[-w:].values
    x = np.arange(len(y))
    try:
        return float(np.polyfit(x, y, 1)[0])
    except:
        return 0.0


# ============================================================
# OU PROCESS
# ============================================================

def compute_ou_halflife(spread):
    if len(spread) < 20:
        return 0, 0, spread.mean(), 0

    x = spread.shift(1).dropna()
    y = spread.iloc[1:]
    if len(x) != len(y):
        return 0, 0, spread.mean(), float(spread.iloc[-1] - spread.mean())

    beta = np.polyfit(x, y, 1)[0]
    mu = spread.mean()
    resid = spread - mu

    if beta <= 0 or beta >= 1:
        return 0, 0, float(mu), float(resid.iloc[-1])

    hl = -np.log(2) / np.log(beta)
    ou_speed = np.log(2) / (hl + _EPS)
    return float(hl), float(ou_speed), float(mu), float(resid.iloc[-1])


# ============================================================
# KALMAN FILTER + OLS
# ============================================================

def ensure_1d(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return np.asarray(s).reshape(-1)

def compute_kalman_hedge_ratio(a, b, state):
    if KalmanFilter is None or len(a) < 10:
        return 0,0,0,state

    y, x = ensure_1d(a), ensure_1d(b)
    mean, cov = (np.zeros(2), np.eye(2)) if state is None else state
    # Update exactly once for the newly observed bar.
    observation = np.array([[x[-1], 1.0]])
    mean, cov = KalmanFilter(n_dim_state=2, n_dim_obs=1).filter_update(
        mean, cov, observation=y[-1:], observation_matrix=observation,
        transition_covariance=np.eye(2) * 0.001,
    )
    beta, intercept = np.asarray(mean)
    residuals = y - (beta * x + intercept)
    z = (residuals[-1] - residuals.mean()) / (residuals.std() + _EPS)
    return float(beta), float(residuals[-1]), float(z), (mean, cov)


def compute_ols_beta_resid(a,b,w):
    if len(a) < w: return 0,0,0
    y=a.iloc[-w:]; x=b.iloc[-w:]
    xc=x-x.mean(); yc=y-y.mean()
    beta = float((xc*yc).sum() / ((xc*xc).sum()+_EPS))
    resid = float(y.iloc[-1] - beta*x.iloc[-1])
    z = float((resid - (y - beta*x).mean()) / ((y-beta*x).std() + _EPS))
    return beta, resid, z


# ============================================================
# COINTEGRATION
# ============================================================

def compute_coint_pvalue(a,b):
    if coint is None: return 1.0
    try:
        _,p,_=coint(a,b)
        return float(p)
    except:
        return 1.0

def compute_adf_pvalue(s):
    if adfuller is None: return 1.0
    try:
        _,p,_,_,_,_ = adfuller(s.dropna(), autolag="AIC")
        return float(p)
    except:
        return 1.0


# ============================================================
# MICROSTRUCTURE
# ============================================================

def parkinson_vol(h,l):
    if len(h) < 2: return 0.0
    rs = (np.log(h/l))**2
    return float(np.sqrt((1/(4*np.log(2))) * rs.mean()))

def garman_klass_vol(o,h,l,c):
    if len(o) < 2: return 0.0
    rs = 0.5*(np.log(h/l))**2 - (2*np.log(2)-1)*(np.log(c/o))**2
    return float(np.sqrt(rs.mean()))

def effective_spread(h,l,c):
    if len(h)==0: return 0.0
    return float((h.iloc[-1]-l.iloc[-1])/(c.iloc[-1]+_EPS))

def vol_signature(p):
    r=_pct_change(p).dropna()
    if len(r)==0: return 0.0
    return float(np.sqrt((r**2).mean()))


# ============================================================
# MAIN FEATURE PIPELINE (FINAL TUNED)
# ============================================================

def compute_all_advanced_features(
    ohlcv,
    A,
    B,
    step,
    kalman_state,
    base_window=30,
    min_periods=60,
    price_type="Adj Close",
    hedge_ratio=1.0,
    intercept=0.0,
):
    """
    Returns: features, new Kalman state, metrics dict
    """

    # Early warm-up. This is deliberately configurable so short validation
    # and test windows can use trailing train context instead of producing
    # all-zero features.
    if step < min_periods:
        return [0]*N_ADVANCED_PAIR_FEATURES, kalman_state, {
            "spread":0,"spread_prev":0,"zscore":0,"zscore_slope":0,
            "spread_velocity":0,"spread_acceleration":0,"spread_jerk":0,
            "ou_halflife":0,"ou_speed":0,"ou_resid":0,"ou_resid_z":0,
            "kf_beta":0,"kf_resid":0,"kf_resid_z":0,
            "ols_beta":0,"ols_resid":0,"ols_resid_z":0,
            "coint_pvalue":0,"adf_pvalue":0,
            "roll_corr":0,"roll_vol":0,
            "liquidity_zscore":0,
            "port_momentum":0,
        }

    # Extract aligned panels
    def cut(k): return ohlcv[k][A].iloc[:step], ohlcv[k][B].iloc[:step]

    openA, openB   = cut("Open")
    highA, highB   = cut("High")
    lowA, lowB     = cut("Low")
    closeA, closeB = cut("Close")
    adjA, adjB     = cut(price_type)
    volA, volB     = cut("Volume")

    # Spread + velocity terms
    spread = adjA - hedge_ratio * adjB - intercept
    ds = spread.diff()
    spread_now = float(spread.iloc[-1])
    spread_vel = float(ds.iloc[-1])
    spread_acc = float(ds.diff().iloc[-1])
    spread_jerk = float(ds.diff().diff().iloc[-1])

    # Zscore + vol
    z10=_rolling_z(spread,10); z20=_rolling_z(spread,20)
    z30=_rolling_z(spread,30); z60=_rolling_z(spread,60)

    vol10=_rolling_vol(spread,10); vol20=_rolling_vol(spread,20)
    vol30=_rolling_vol(spread,30); vol60=_rolling_vol(spread,60)

    # Bollinger
    if len(spread)>=20:
        s20=spread.iloc[-20:]; m20=s20.mean(); s20sd=s20.std()+_EPS
        bb_pos=(spread_now-m20)/(2*s20sd)
        bb_width=(s20.max()-s20.min())/(abs(m20)+_EPS)
        bb_in=1.0 if abs((spread_now-m20)/s20sd)<=2 else 0.0
    else:
        bb_pos=bb_width=bb_in=0.0

    # Returns
    rA=_pct_change(adjA); rB=_pct_change(adjB); rS=(spread.diff() / (adjA + abs(hedge_ratio) * adjB).shift(1)).fillna(0)

    windows=[5,10,20,30,60,90,120,180,252]

    # Spread return stats
    spread_stats=[]
    for w in windows:
        if len(rS)<w:
            spread_stats += [0,0,0]
        else:
            chunk = rS.iloc[-w:]
            m = chunk.mean()
            sd = chunk.std()+_EPS
            sh = m/sd
            spread_stats += [float(m),float(sd),float(sh)]

    # Trend features for A, B
    trend_A=[]; trend_B=[]
    for w in windows:
        if len(adjA)>=w:
            trend_A += [_poly_slope(adjA,w),
                        float(adjA.iloc[-1]/(adjA.iloc[-w]+_EPS)-1)]
        else:
            trend_A += [0,0]

        if len(adjB)>=w:
            trend_B += [_poly_slope(adjB,w),
                        float(adjB.iloc[-1]/(adjB.iloc[-w]+_EPS)-1)]
        else:
            trend_B += [0,0]

    # Correlations
    corrs = [_rolling_corr(rA,rB,w) for w in windows]

    # OU
    ou_hl, ou_speed, ou_mu, ou_resid = compute_ou_halflife(spread)
    ou_resid_z = _rolling_z(spread - ou_mu, 30)

    # Kalman / OLS
    k_beta, k_resid, k_resid_z, new_k = compute_kalman_hedge_ratio(adjA,adjB,kalman_state)
    ols_beta, ols_resid, ols_resid_z = compute_ols_beta_resid(adjA,adjB,base_window)

    # Microstructure
    parkA=parkinson_vol(highA,lowA); parkB=parkinson_vol(highB,lowB)
    gkA=garman_klass_vol(openA,highA,lowA,closeA)
    gkB=garman_klass_vol(openB,highB,lowB,closeB)
    effA=effective_spread(highA,lowA,closeA)
    effB=effective_spread(highB,lowB,closeB)
    sigA=vol_signature(adjA); sigB=vol_signature(adjB)

    # Liquidity (8×2 = 16)
    vol_feats=[]
    for v,p in [(volA,adjA),(volB,adjB)]:
        if len(v)<20:
            vol_feats += [0]*8
        else:
            v20=v.iloc[-20:]
            lvl=float(v.iloc[-1]); m=float(v20.mean())
            sd=float(v20.std()+_EPS)
            z=float((lvl-m)/sd)
            cv=sd/(m+_EPS)
            ac=float(v20.autocorr(1)) if len(v20)>1 else 0
            slope=_poly_slope(v,20)
            rtmp=_pct_change(p).iloc[-20:]
            dollar = v20 * p.iloc[-20:]
            amihud=float(np.mean(np.abs(rtmp)/(dollar+_EPS)))
            vol_feats += [lvl,m,sd,z,cv,ac,slope,amihud]

    # Cointegration + ADF
    coint_p = compute_coint_pvalue(adjA,adjB)
    adf_p   = compute_adf_pvalue(spread)
    coint_strength = 1-coint_p
    adf_strength   = -np.log10(adf_p+_EPS)

    # Technicals (12)
    def _rsi(x,w=14):
        if len(x)<w+1: return 0.0
        d=x.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
        rs=up.rolling(w).mean()/(dn.rolling(w).mean()+_EPS)
        return float(100 - 100/(1+rs.iloc[-1]))

    rsiA_14=_rsi(adjA,14); rsiA_30=_rsi(adjA,30)

    def _macd(x):
        if len(x)<35: return 0,0
        ef=x.ewm(span=12).mean()
        es=x.ewm(span=26).mean()
        macd=ef-es
        sig=macd.ewm(span=9).mean()
        return float(macd.iloc[-1]), float(sig.iloc[-1])

    macdA,macdA_sig=_macd(adjA)
    macdS,macdS_sig=_macd(spread)

    def _stoch(h,l,c,w=14):
        if len(c)<w: return 0,0
        hh=h.iloc[-w:].max(); ll=l.iloc[-w:].min()
        k=float(100*(c.iloc[-1]-ll)/(hh-ll+_EPS))
        return k,k

    stochK,stochD = _stoch(highA,lowA,closeA)

    def _mfi(h,l,c,v,w=14):
        if len(c)<w+1: return 0
        tp=(h+l+c)/3; mf=tp*v
        change=tp.diff().iloc[-w:]
        flow=mf.iloc[-w:]
        mr=flow.where(change > 0, 0).sum() / (flow.where(change < 0, 0).sum()+_EPS)
        return float(100 - 100/(1+mr))

    mfiA=_mfi(highA,lowA,closeA,volA)

    def _obv(c,v):
        if len(c)<2: return 0.0
        dir=np.sign(c.diff().fillna(0))
        obv=(dir*v).cumsum()
        return float(obv.iloc[-1]/(v.mean()+_EPS))

    obvA=_obv(closeA,volA)
    spread_rsi=_rsi(spread,14)

    # Tail risk (8)
    if len(rS)>=60:
        r60=rS.iloc[-60:]
        skew=float(r60.skew())
        kurt=float(r60.kurt())
        dd=float(np.sqrt(np.mean(np.minimum(r60,0)**2)))
        semi=float(np.mean(np.minimum(r60,0)**2))
        var95=np.quantile(r60,0.05)
        c95=float(r60[r60<=var95].mean())
        var99=np.quantile(r60,0.01)
        c99=float(r60[r60<=var99].mean())
    else:
        skew=kurt=dd=semi=var95=c95=var99=c99=0.0


    # ============================================================
    # FEATURES (FINAL PACK)
    # ============================================================

    feats = []
    feats += [spread_now, spread_vel, spread_acc, spread_jerk, abs(spread_now), np.sign(spread_now)]
    feats += [z10,z20,z30,z60]
    feats += [vol10,vol20,vol30,vol60]
    feats += [bb_pos,bb_width,bb_in]
    feats += spread_stats
    feats += trend_A
    feats += trend_B
    feats += corrs
    feats += [ou_hl,ou_speed,ou_mu,ou_resid,ou_resid_z,spread_now-ou_mu]
    feats += [k_beta,k_resid,k_resid_z,ols_beta,ols_resid,ols_resid_z,k_beta-ols_beta,k_resid-ols_resid]
    feats += [parkA,parkB,gkA,gkB,effA,effB,sigA,sigB]
    feats += vol_feats
    feats += [coint_p,adf_p,coint_strength,adf_strength]
    feats += [
        rsiA_14,rsiA_30,
        stochK,stochD,
        macdA,macdA_sig,
        mfiA,obvA,
        spread_rsi,
        macdS,macdS_sig,
        spread_rsi*rsiA_14
    ]
    feats += [skew,kurt,dd,semi,var95,c95,var99,c99]


    if len(feats) != N_ADVANCED_PAIR_FEATURES:
        raise ValueError(f"Expected {N_ADVANCED_PAIR_FEATURES} features, got {len(feats)}")
    feats = np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


    # ============================================================
    # METRICS FOR REWARD FUNCTION
    # ============================================================

    metrics = {
        "spread": spread_now,
        "spread_prev": float(spread.iloc[-2]) if len(spread)>=2 else spread_now,
        "zscore": z30,
        "zscore_slope": _poly_slope(spread,30),
        "spread_velocity": spread_vel,
        "spread_acceleration": spread_acc,
        "spread_jerk": spread_jerk,
        "ou_halflife": ou_hl,
        "ou_speed": ou_speed,
        "ou_resid": ou_resid,
        "ou_resid_z": ou_resid_z,
        "kf_beta": k_beta,
        "kf_resid": k_resid,
        "kf_resid_z": k_resid_z,
        "ols_beta": ols_beta,
        "ols_resid": ols_resid,
        "ols_resid_z": ols_resid_z,
        "coint_pvalue": coint_p,
        "adf_pvalue": adf_p,
        "roll_corr": _rolling_corr(rA,rB,30),
        "roll_vol": vol30,
        "liquidity_zscore": vol_feats[3] if len(vol_feats)>=4 else 0.0,
        "port_momentum": float(rS.iloc[-1]) if len(rS)>0 else 0.0,
    }

    return feats, new_k, metrics

