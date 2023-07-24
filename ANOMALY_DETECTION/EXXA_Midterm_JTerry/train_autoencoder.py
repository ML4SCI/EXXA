import argparse
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, progress
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, "./")
# from utils.data_utils import
from models.autoencoder import Autoencoder
from utils.data_utils import prepare_datasets
from utils.globals import data_path, watt_to_jansky

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
        prog="GAN anomaly",
        description="Trains an autoencoder to recreate Kerplerian spectra",
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_eps", type=float, default=1e-6, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Learning rate decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout fraction")
    parser.add_argument("--mlp_layer_dim", type=int, default=48, help="Dimension")
    parser.add_argument(
        "--num_mlp_layers", type=int, default=5, help="Number of discriminator layers"
    )

    parser.add_argument(
        "--use_batchnorm", type=int, default=0, help="Use batch normalization for generator?"
    )
    parser.add_argument("--leaky_relu_frac", type=float, default=0.2, help="Leaky relu value")

    parser.add_argument(
        "--weight_init", type=str, default="xavier", help="Weight initialization method"
    )

    parser.add_argument("--activation", type=str, default="gelu", help="Activaiton function")

    parser.add_argument(
        "--add_noise", type=int, default=0, help="Add noise to the input spectrum"
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
        default="autoencoder_anomaly",
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

    # parser.add_argument(
    #     "--use_self_attention", type=int, default=0, help="Use self-attention in model"
    # )

    parser.add_argument(
        "--max_seq_length", type=float, default=101, help="Maximum length of spectrum"
    )

    parser.add_argument("--latent_dim", type=int, default=48, help="Generator latent dimension")
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

    args = parser.parse_args()

    # data_path = args.data_path
    wandb_project_name = args.wandb_project_name
    model_save_path = args.model_save_path
    scale_data = args.scale_data
    num_workers = args.num_workers
    wandb_sweep = bool(args.wandb_sweep)
    max_seq_length = args.max_seq_length
    log_wandb = bool(args.log_wandb)
    line_index = args.line_index
    max_v = args.max_v
    accelerator_name = args.accelerator_name

    # wandb_project_name += f"_{method}"

    # get hyperparameters as a dictionary
    model_hparams = vars(args)

    model_hparams["input_dim"] = max_seq_length
    model_hparams["output_dim"] = max_seq_length

    if model_hparams["scale_data"] == 0.0:
        model_hparams["scale_data"] = 1.0 / watt_to_jansky
        scale_data = model_hparams["scale_data"]

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
    train_data, val_data, test_data = prepare_datasets(
        data_path=data_path,
        scale_data=scale_data,
        max_seq_length=max_seq_length,
        val_split=validation_split,
        test_split=test_split,
        line_index=line_index,
        max_v=max_v,
        accelerator_name=accelerator_name,
    )

    # get data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

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
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False,
                mode="min",
                monitor="val_loss",
                save_top_k=1,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
                dirpath=f"{model_save_path}Checkpoints/{run_name}/",
                filename=f"final_model_checkpoint_{run_name}",
            ),
            LearningRateMonitor("epoch"),
            progress.TQDMProgressBar(refresh_rate=1),
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
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
    model = Autoencoder(
        latent_dim=model_hparams["latent_dim"],
        max_seq_length=max_seq_length,
        lr=model_hparams["lr"],
        adam_eps=model_hparams["adam_eps"],
        weight_decay=model_hparams["weight_decay"],
        weight_init=model_hparams["weight_init"],
        use_batchnorm=model_hparams["use_batchnorm"],
        num_mlp_layers=model_hparams["num_mlp_layers"],
        mlp_layer_dim=model_hparams["mlp_layer_dim"],
        activation=model_hparams["activation"],
        leaky_relu_frac=model_hparams["leaky_relu_frac"],
        dropout=model_hparams["dropout"],
        add_noise=bool(model_hparams["add_noise"]),
    )
    model.to(device)

    # fit the model
    print("Training")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

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
    train_data, val_data, train_loader, val_loader = None, None, None, None

    # Test the model
    ## Get test metrics
    print("Testing model")
    print(trainer.test(model, test_loader))

    # access the 'test_loss' (list of dict)
    # loss_value = test_loss[0]["test_g_loss"]

    # use loss_value value for  wandb.log
    # wandb.log({"final_loss": loss_value})

    wandb.finish()
