import numpy as np

# ======================================================================
# Simplified Reward
# ======================================================================
def reward_style_simple(
    portfolio_value,
    prev_portfolio_value,
    position_size,
    spread,
    prev_spread,
    zscore,
    port_vol=0.0,

    lam_ret=1.0,
    lam_mr=0.5,
    lam_pos=0.05,
    lam_z=0.05,
    lam_vol=0.05,

    reward_clip=5.0,
):
    """Stable minimalist reward for inverse pairs trading."""

    # Log return
    raw_ret = (portfolio_value - prev_portfolio_value) / (abs(prev_portfolio_value) + 1e-12)
    log_ret = np.clip(raw_ret, -0.10, 0.10)

    # Mean reversion bonus
    mr_bonus = lam_mr * (abs(prev_spread) - abs(spread))

    # Penalties
    pos_penalty = lam_pos * (position_size ** 2)
    z_penalty   = lam_z * abs(zscore)
    vol_penalty = lam_vol * abs(log_ret) * (1 + port_vol)

    reward = (
        lam_ret * log_ret
        + mr_bonus
        - pos_penalty
        - z_penalty
        - vol_penalty
    )

    return float(np.clip(reward, -reward_clip, reward_clip))