import argparse
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, progress
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, "./")
# from utils.data_utils import
from models.multiloader_transformer import MultiloaderTransformerSeq2Seq
from utils.data_utils import make_multiple_datasets
from utils.globals import data_path, watt_to_jansky
from utils.model_utils import save_model_hparams

### Set seed for reproducibility
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)


def check_and_make_dir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    print("Parsing arguments")

    parser = argparse.ArgumentParser(
        prog="Transformer anomaly",
        description="Trains a transformer to recreate Kerplerian spectra",
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_eps", type=float, default=1e-6, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Learning rate decay")
    parser.add_argument(
        "--pct_start",
        type=float,
        default=0.1,
        help="Run percentage to increase lr (cycle scheduler)",
    )
    parser.add_argument(
        "--max_lr_factor", type=float, default=1.5, help="Maximum learning rate (cycle scheduler)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Learning rate decay factor (exp and step scheduler)",
    )
    parser.add_argument(
        "--eta_min", type=float, default=1e-9, help="Minimum learning rate (cosine scheduler)"
    )
    parser.add_argument(
        "--pos_enc_scale",
        type=float,
        default=1.0,
        help="Factor by which to scale the positional encodings",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout fraction")
    parser.add_argument(
        "--pf_dim", type=int, default=48, help="position-wise feed-forward dimension"
    )
    parser.add_argument("--hid_dim", type=int, default=4, help="Transformer hidden dimension")
    parser.add_argument(
        "--n_layers", type=int, default=3, help="Number of transformer encoding layers"
    )

    parser.add_argument(
        "--n_heads", type=int, default=2, help="Number of transformer encoding layers"
    )

    parser.add_argument("--activation", type=str, default="gelu", help="Activaiton function")

    parser.add_argument(
        "--add_noise", type=int, default=0, help="Add noise to the input spectrum"
    )
    parser.add_argument(
        "--trg_eq_zero", type=int, default=1, help="Make the target spectrum all zeros"
    )
    parser.add_argument(
        "--scheduler_name",
        type=str,
        default="cosine",
        help="Learning rate scheduler name (cosine, step, exp, cycle, none)",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./Data/Keplerian/",
        help="Directory where data is stored",
    )

    parser.add_argument(
        "--log_wandb",
        type=int,
        default=1,
        help="Use wandb logger",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="multiloader_transformer_anomaly",
        help="Name of wandb project",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./trained_models/",
        help="Directory to save the models",
    )

    parser.add_argument(
        "--scale_data", type=float, default=-1.0, help="Factor by which to scale up data"
    )

    parser.add_argument(
        "--one_per_run", type=int, default=1, help="Only use a single dump file per simulation"
    )

    # parser.add_argument(
    #     "--padded_spectrum", type=int, default=0, help="Pad spectrum with inifinities"
    # )

    parser.add_argument(
        "--line_index",
        type=int,
        default=1,
        help="Emission line index (0 = 1-0, 1 = 2-1, 2 = 3-2)",
    )
    parser.add_argument(
        "--max_v",
        type=float,
        default=5.0,
        help="Maximum (absolute value) velocity channel relative to systemic",
    )

    parser.add_argument(
        "--sub_cont",
        type=int,
        default=0,
        help="Subtract continuum",
    )

    parser.add_argument(
        "--wandb_sweep",
        type=int,
        default=0,
        help="Do a WandB sweep",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--accelerator_name",
        type=str,
        default="mps",
        help="Type of accelerator to use (if available)",
    )

    parser.add_argument(
        "--ignore_zeros",
        type=int,
        default=1,
        help="Ignore spectra that are only zeros",
    )
    parser.add_argument(
        "--parameter_path",
        type=str,
        default="./Parameters/",
        help="Where to save model parameters",
    )

    args = parser.parse_args()

    # data_path = args.data_path
    wandb_project_name = args.wandb_project_name
    model_save_path = args.model_save_path
    scale_data = args.scale_data
    num_workers = args.num_workers
    wandb_sweep = bool(args.wandb_sweep)
    # max_seq_length = args.max_seq_length
    log_wandb = bool(args.log_wandb)
    line_index = args.line_index
    max_v = args.max_v
    one_per_run = bool(args.one_per_run)
    accelerator_name = args.accelerator_name
    ignore_zeros = bool(args.ignore_zeros)

    # wandb_project_name += f"_{method}"

    # get hyperparameters as a dictionary
    model_hparams = vars(args)

    # if bool(args.padded_spectrum):
    #     wandb_project_name = f"padded_{wandb_project_name}"
    #     model_hparams["wandb_project_name"] = wandb_project_name

    model_hparams["input_dim"] = 1
    model_hparams["output_dim"] = 1

    if model_hparams["scale_data"] == 0.0:
        model_hparams["scale_data"] = 1.0 / watt_to_jansky
        scale_data = model_hparams["scale_data"]

    # number of heads has to be factor of hidden_dim
    if model_hparams["hid_dim"] % model_hparams["n_heads"] != 0:
        if model_hparams["hid_dim"] < model_hparams["n_heads"]:
            model_hparams["n_heads"] = model_hparams["hid_dim"]
        else:
            # model_hparams["num_heads"] = max([1, model_hparams["hidden_dim"] // model_hparams["num_heads"]])
            while model_hparams["hid_dim"] % model_hparams["n_heads"] != 0:
                model_hparams["n_heads"] -= 1

    # del model_hparams["data_path"]
    # # del model_hparams["wandb_project_name"]
    # del model_hparams["model_save_path"]
    # del model_hparams["num_workers"]
    # del model_hparams["wandb_sweep"]
    # del model_hparams["log_wandb"]
    # del model_hparams["accelerator_name"]
    keys_to_delete = [
        "data_path",
        "model_save_path",
        "num_workers",
        "wandb_sweep",
        "log_wandb",
        "dirty_ext",
        "accelerator_name",
    ]

    for key in keys_to_delete:
        if key in model_hparams:
            del model_hparams[key]

    validation_split = 0.2
    test_split = 0.2
    batch_size = 64
    num_epochs = 50
    include_noise = True

    check_and_make_dir(model_save_path)
    check_and_make_dir(f"{model_save_path}Checkpoints/")

    print("Getting data")
    # get datasets
    train_data, val_data, test_data = make_multiple_datasets(
        data_path,
        scale_data=scale_data,
        val_split=validation_split,
        test_split=test_split,
        line_index=line_index,
        max_v=max_v,
        one_per_run=one_per_run,
        accelerator_name=accelerator_name,
        ignore_zeros=ignore_zeros,
        sub_cont=bool(model_hparams["sub_cont"]),
    )

    run_order = sorted(train_data.keys())
    # get data loaders
    train_loaders = {
        run: DataLoader(
            train_data[run], batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        for run in run_order
    }
    # val_loaders = dict((run, DataLoader(val_data[run],
    #                                batch_size=batch_size,
    #                                num_workers=num_workers))
    #                    for run in run_order)
    # test_loaders = dict((run, DataLoader(test_data[run],
    #                                 batch_size=batch_size,
    #                                 num_workers=num_workers))
    #                     for run in run_order)

    # train_loaders = [DataLoader(
    #     train_data[run],
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers
    # ) for run in run_order]

    run_order = sorted(test_data.keys())
    val_loaders = [
        DataLoader(val_data[run], batch_size=batch_size, num_workers=num_workers)
        for run in run_order
    ]
    run_order = sorted(test_data.keys())
    test_loaders = [
        DataLoader(test_data[run], batch_size=batch_size, num_workers=num_workers)
        for run in run_order
    ]

    combined_train_loader = CombinedLoader(train_loaders, mode="max_size_cycle")
    combined_test_loader = CombinedLoader(test_loaders, mode="sequential")
    combined_val_loader = CombinedLoader(val_loaders, mode="sequential")

    print("Setting up WandB")

    # set up wandb logger
    os.environ["WANDB_API_KEY"] = "199115cad71655dbb5640225359e90bc0a91bcca"
    # wandb entity
    entity = "chlab"
    logger_kwargs = {
        "resume": "allow",
        "config": model_hparams,
    }

    # if log_wandb:
    logger = WandbLogger(project=wandb_project_name, entity=entity, **logger_kwargs)
    run_name = logger.experiment.name
    check_and_make_dir(f"{model_save_path}Checkpoints/{run_name}/")

    print("Setting up run")

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    if accelerator_name == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif accelerator_name == "cuda:0":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    #### necessary for newer PTL versions
    devices = 1
    accelerator = "gpu" if devices == 1 else "cpu"

    # make the trainer
    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=num_epochs,
        log_every_n_steps=1,
        logger=logger,
        # reload_dataloaders_every_epoch=True,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False,
                mode="min",
                monitor="avg_val_loss",
                save_top_k=1,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
                dirpath=f"{model_save_path}Checkpoints/{run_name}/",
                filename=f"final_model_checkpoint_{run_name}",
            ),
            LearningRateMonitor("epoch"),
            progress.TQDMProgressBar(refresh_rate=1),
            EarlyStopping(
                monitor="avg_val_loss",
                min_delta=0,
                patience=10,
                verbose=False,
                mode="min",
            ),
        ],
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    print("Creating model")
    # get the model
    model = MultiloaderTransformerSeq2Seq(
        input_dim=model_hparams["input_dim"],
        output_dim=model_hparams["output_dim"],
        n_heads=model_hparams["n_heads"],
        n_layers=model_hparams["n_layers"],
        hid_dim=model_hparams["hid_dim"],
        pf_dim=model_hparams["pf_dim"],
        lr=model_hparams["lr"],
        adam_eps=model_hparams["adam_eps"],
        weight_decay=model_hparams["weight_decay"],
        activation=model_hparams["activation"],
        dropout=model_hparams["dropout"],
        add_noise=bool(model_hparams["add_noise"]),
        trg_eq_zero=bool(model_hparams["trg_eq_zero"]),
        device=device,
        max_v=model_hparams["max_v"],
        line_index=model_hparams["line_index"],
        data_length=len(train_data),
        max_lr_factor=model_hparams["max_lr_factor"],
        pct_start=model_hparams["pct_start"],
        pos_enc_scale=model_hparams["pos_enc_scale"],
        gamma=model_hparams["gamma"],
        eta_min=model_hparams["eta_min"],
        scheduler_name=model_hparams["scheduler_name"],
    )
    model.to(device)

    # fit the model
    print("Training")
    trainer.fit(
        model,
        train_dataloaders=combined_train_loader,
        val_dataloaders=combined_val_loader,
    )

    # save the model
    print("Saving final model")
    # torch.save(model, f"{model_save_path}trained_embedder_{run_name}.pyt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": model.optimizer.state_dict(),
        },
        f"{model_save_path}final_model_{run_name}.pyt",
    )

    # clear memory
    train_data, val_data, train_loaders, val_loaders = None, None, None, None
    combined_train_loader, combined_val_loader = None, None

    # Test the model
    ## Get test metrics
    print("Testing model")
    print(trainer.test(model, combined_test_loader))

    save_model_hparams(
        model_hparams, run_name, "multiloader_transformer", hparam_path=args.parameter_path
    )

    wandb.finish()
