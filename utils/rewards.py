import numpy as np

def reward_style_3(
    portfolio_value,
    prev_portfolio_value,
    position_size,

    spread,
    prev_spread,
    zscore,
    zscore_slope,
    spread_velocity,
    spread_acceleration,
    spread_jerk,

    ou_halflife,
    ou_speed,
    ou_resid,
    ou_resid_z,

    kf_beta,
    kf_resid,
    kf_resid_z,
    ols_beta,
    ols_resid,
    ols_resid_z,

    coint_pvalue,
    adf_pvalue,

    roll_corr,
    roll_vol,
    port_vol,

    liquidity_zscore,

    port_momentum,

    # Hyperparams
    lam_ret=1.0,
    lam_vol=0.20,
    lam_corr=0.10,
    lam_rollvol=0.08,
    lam_pos=0.05,
    lam_mr=0.25,
    lam_z=0.08,
    lam_trend=0.20,
    lam_kalman=0.15,
    lam_ols=0.15,
    lam_coint=0.20,
    lam_liq=0.05,
    lam_mom=0.10,
    lam_stab=0.12,

    # NEW: clipping value from config
    reward_clip=20.0,
):
    """Numerically-stable reward, fully tied to config, with correct clipping."""

    # =============================
    # Clip unstable inputs
    # =============================
    position_size = np.clip(position_size, -5.0, 5.0)
    zscore        = np.clip(zscore, -10, 10)
    zscore_slope  = np.clip(zscore_slope, -5, 5)

    spread_velocity      = np.clip(spread_velocity, -5, 5)
    spread_acceleration  = np.clip(spread_acceleration, -5, 5)
    spread_jerk          = np.clip(spread_jerk, -5, 5)

    ou_resid    = np.clip(ou_resid, -5, 5)
    ou_resid_z  = np.clip(ou_resid_z, -10, 10)
    ou_speed    = np.clip(ou_speed, -5, 5)

    kf_resid    = np.clip(kf_resid, -5, 5)
    kf_resid_z  = np.clip(kf_resid_z, -10, 10)
    ols_resid   = np.clip(ols_resid, -5, 5)
    ols_resid_z = np.clip(ols_resid_z, -10, 10)

    port_momentum     = np.clip(port_momentum, -5, 5)
    roll_vol          = np.clip(roll_vol, 0, 5)
    liquidity_zscore  = np.clip(liquidity_zscore, -10, 10)

    # =============================
    # Log return
    # =============================
    # SAFER return signal
    raw_ret = (portfolio_value - prev_portfolio_value) / (abs(prev_portfolio_value) + 1e-12)

    # Fully clipped and PPO-stable
    log_ret = np.clip(raw_ret, -0.10, 0.10)

    # =============================
    # Reward components
    # =============================
    vol_penalty = lam_vol * abs(log_ret) * (1 + port_vol)
    pos_penalty = lam_pos * (position_size ** 2)

    mr_bonus = lam_mr * (abs(prev_spread) - abs(spread))
    ou_strength_bonus = lam_mr * np.clip(1 / (ou_halflife + 1e-6), -1, 1)
    ou_resid_penalty = lam_mr * abs(ou_resid_z)
    ou_speed_bonus = lam_mr * np.tanh(ou_speed)
    ou_resid_shrink_bonus = lam_mr * (abs(prev_spread) - abs(ou_resid))

    trend_penalty = lam_trend * np.sign(spread) * (spread - prev_spread)
    speed_penalty = lam_trend * (
        abs(spread_velocity)
        + 0.5 * abs(spread_acceleration)
        + 0.25 * abs(spread_jerk)
    )

    roll_vol_penalty = lam_rollvol * roll_vol
    corr_bonus = lam_corr * abs(roll_corr)

    kalman_penalty = lam_kalman * (
        abs(kf_resid_z)
        + abs(kf_resid)
        + 0.5 * abs(kf_beta - ols_beta)
    )

    ols_penalty = lam_ols * (abs(ols_resid_z) + 0.5 * abs(ols_resid))

    coint_bonus = lam_coint * (1 - coint_pvalue)
    adf_bonus   = lam_coint * (1 - adf_pvalue)

    liq_penalty = lam_liq * abs(liquidity_zscore)
    momentum_penalty = lam_mom * abs(port_momentum)

    stability_penalty = lam_stab * (
        abs(zscore_slope)
        + 0.5 * abs(spread_acceleration)
        + 0.25 * abs(spread_jerk)
    )
    z_penalty = lam_z * abs(zscore)

    # =============================
    # Combine reward
    # =============================
    reward = (
        lam_ret * log_ret
        - vol_penalty
        - pos_penalty
        + mr_bonus
        + ou_strength_bonus
        - ou_resid_penalty
        + ou_speed_bonus
        + ou_resid_shrink_bonus
        - trend_penalty
        - speed_penalty
        - kalman_penalty
        - ols_penalty
        + corr_bonus
        - roll_vol_penalty
        + coint_bonus
        + adf_bonus
        - liq_penalty
        - momentum_penalty
        - stability_penalty
        - z_penalty
    )

    # =============================
    # ✔ FINAL CONFIG-BASED CLIP
    # =============================
    reward = float(np.clip(reward, -reward_clip, reward_clip))

    return reward

