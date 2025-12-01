import time
import os
import yaml
from dotenv import load_dotenv

from live.alpaca_stream import AlpacaStream
from live.alpaca_order_executor import AlpacaOrderExecutor
from live.live_env import LiveInversePairsEnv
from live.portfolio_tracker import PortfolioTracker

from agents.ppo_agent import PPOAgent


def main():

    load_dotenv()

    # Load inverse pairs
    with open("config/pairs.yaml", "r") as f:
        pairs = yaml.safe_load(f)["inverse_pairs"][:10]   # top 10

    # Load config
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # RL agent
    agent = PPOAgent(model_path="models/ppo_inverse_pairs.zip")

    # Live environment identical to training environment
    env = LiveInversePairsEnv(
        pairs=pairs,
        window_size=cfg["environment"]["window_size"],
        kelly_fraction=cfg["environment"]["kelly_fraction"]
    )

    stream = AlpacaStream()
    executor = AlpacaOrderExecutor()
    tracker = PortfolioTracker()

    tickers = sorted(list(set(x for pair in pairs for x in pair)))

    print(" Live trading system started!")

    while True:

        prices = stream.get_latest_prices(tickers)
        if not prices:
            print("No price data yet.")
            time.sleep(60)
            continue

        env.update_prices(prices)

        obs = env.get_observation()
        if obs is None:
            print("Waiting for window to fill...")
            time.sleep(60)
            continue

        action = agent.predict(obs)

        env.decode_and_apply_action(action, prices)
        equity = env.update_portfolio_value(prices)

        print(f"Equity: {equity:.2f}")

        # Execute all open positions
        for (a, b), pos in env.positions.items():

            # Direction: 1 = Long A Short B
            #            -1 = Short A Long B
            d = pos["direction"]

            side_a = "buy" if d == 1 else "sell"
            side_b = "sell" if d == 1 else "buy"

            executor.submit_order(a, pos["size_a"], side_a)
            executor.submit_order(b, pos["size_b"], side_b)

            tracker.log_trade(a, side_a, pos["size_a"], prices[a])
            tracker.log_trade(b, side_b, pos["size_b"], prices[b])

        time.sleep(60)


if __name__ == "__main__":
    main()
