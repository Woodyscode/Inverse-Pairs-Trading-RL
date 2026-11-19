"""The historical PPO checkpoints do not contain a deployable live execution policy."""


def main():
    raise SystemExit(
        "Live execution is disabled. Use main_train.py for the audited research simulator."
    )


if __name__ == "__main__":
    main()
