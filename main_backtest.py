"""Run the walk-forward z-score benchmark using the research execution model."""
import argparse

from main_train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    main(parser.parse_args().config, benchmark_only=True)
