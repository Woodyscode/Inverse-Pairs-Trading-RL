import numpy as np

# ======================================================================
#                  FINAL PRODUCTION PNL REWARD FUNCTION
# ======================================================================
def reward_pnl(
    portfolio_value,
    prev_portfolio_value,
    position_size,
    reward_clip=5.0,
):
    """
    Pure PnL-based reward for RL trading.
    Handles multi-strategy (traditional + inverse) automatically.
    
    Args:
        portfolio_value: current portfolio value
        prev_portfolio_value: previous portfolio value
        position_size: total exposure (0–max_total_exposure)
        reward_clip: optional clipping for PPO stability
        
    Returns:
        scalar reward
    """

    # ------------------------------------------------------------------
    # 1. Immediate portfolio return
    # ------------------------------------------------------------------
    if prev_portfolio_value <= 0:
        pnl_ret = 0.0
    else:
        pnl_ret = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

    # ------------------------------------------------------------------
    # 2. Tiny position penalty (encourages precision, optional)
    # ------------------------------------------------------------------
    pos_penalty = 0.01 * position_size  # very small

    reward = pnl_ret - pos_penalty

    # ------------------------------------------------------------------
    # 3. Clip to keep PPO stable
    # ------------------------------------------------------------------
    reward = float(np.clip(reward, -reward_clip, reward_clip))

    return reward
