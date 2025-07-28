SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"min": 0.001, "max": 0.01},
        "weight_decay": {"min": 0.0001, "max": 0.0005},
    },
}

CHANNEL_SUBSETS = [
    list(range(30, 36)),
    list(range(36, 42)),
    list(range(42, 48)),
    list(range(48, 54)),
    list(range(54, 60)),
    list(range(60, 66)),
    list(range(66, 72)),
    list(range(72, 78)),
    list(range(78, 84)),
    list(range(84, 90)),
    list(range(90, 96)),
    list(range(96, 101)),
]
