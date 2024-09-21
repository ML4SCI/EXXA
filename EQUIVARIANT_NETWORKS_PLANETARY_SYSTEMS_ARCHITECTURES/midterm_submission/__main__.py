import wandb
from dataset import PlanetaryDataset
from utils import active_learning
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomAffine, ToTensor
import argparse
from rl_agent import run_multi_agent_system

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run active learning with MAS or RL approach."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["simple", "mas"],
        required=True,
        help="Method to use: simple or mas",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["EquivariantHybridModel", "E2SteerableCNN"],
        required=True,
        help="Model to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--n_agents", type=int, default=3, help="Number of agents to use for MAS"
    )
    args = parser.parse_args()

    channel_subsets = [
        list(range(30, 42)),
        list(range(42, 54)),
        list(range(54, 66)),
        list(range(66, 78)),
        list(range(78, 90)),
        list(range(90, 101)),
    ]

    dataset = PlanetaryDataset(
        data_dir="/content/drive/MyDrive/Kinematic_Data/Train_Clean",
        csv_file="/content/drive/MyDrive/Kinematic_Data/train_info.csv",
        channels=[],
        transform=Compose(
            [
                RandomHorizontalFlip(p=0.5),
                RandomAffine(degrees=0, translate=(0.1, 0.1)),
                ToTensor(),
            ]
        ),
    )

    wandb.login()
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "train_loss", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 0.0001, "max": 0.01},
            "weight_decay": {"min": 0.00001, "max": 0.001},
        },
    }
    if args.method == "simple":
        sweep_id = wandb.sweep(sweep_config, project="eq_colab_2")
        wandb.agent(
            sweep_id,
            lambda: active_learning(
                args.model,
                channel_subsets,
                dataset,
                initial_subset_size=5,
                n_iterations=5,
                epochs=args.epochs,
            ),
        )
    else:
        state_size = len(channel_subsets)
        action_size = len(channel_subsets)
        run_multi_agent_system(
            args.model,
            dataset,
            channel_subsets,
            state_size,
            action_size,
            n_agents=args.n_agents,
            n_iterations=5,
            epochs=args.epochs,
        )

    wandb.finish()
