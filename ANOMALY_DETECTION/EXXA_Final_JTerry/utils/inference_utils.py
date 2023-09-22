import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from astropy.io import fits
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from matplotlib import rc as mplrc
from matplotlib.colors import LogNorm

from utils.data_utils import (
    cycle_spectrum,
    fill_spectrum,
    normalize_spectrum,
    turn_data_cube_into_spectra,
)
from utils.globals import (
    all_model_hparams,
    channel_limits,
    labels,
    legends,
    lw,
    model_types,
    systemic_channels,
    ticks,
    titles,
    translations,
)
from utils.model_utils import load_trained_model


def anomaly_inference(
    model: torch.nn.Module,
    spectra_input: np.ndarray,
    vel_input: np.ndarray,
    keys: dict,
    data: np.ndarray,
    device: torch.device,
    inference_batches: int = 0,
    pad_value: float | None = None,
) -> (np.ndarray, np.ndarray, int, int):
    # inference on entire thing
    if inference_batches == 0:
        print("Inference")
        spec_tensor = torch.from_numpy(spectra_input).float().to(device)
        pred_specs = model(spec_tensor).detach().cpu().numpy()
    # save memory during inference
    else:
        print("Inference")
        # do inference over several batches
        pred_specs = np.zeros(spectra_input.shape, dtype=np.float32)
        for i in range(0, len(pred_specs), inference_batches):
            spec_tensor = (
                torch.from_numpy(spectra_input[i : i + inference_batches]).float().to(device)
            )
            pred_specs[i : i + inference_batches] = model(spec_tensor).detach().cpu().numpy()

    print("Making error map")
    min_mse = 1e6
    max_mse = 0
    best_index = 0
    worst_index = 0

    anomaly_map = np.zeros(data.shape[1:], dtype=np.float32)
    vel_err_map = np.zeros(data.shape[1:], dtype=np.float32)

    # get reconstruction MSE and put into map
    for i in keys:
        x, y = keys[i][0], keys[i][1]
        spec = spectra_input[i]
        pred_spec = pred_specs[i]
        if pad_value is not None:
            good = np.where(spec != pad_value)
            spec = spec[good]
            pred_spec = pred_spec[good]
        mse_array = (spec - pred_spec) ** 2.0
        mse = np.mean(mse_array)
        max_err_index = np.argmax(mse_array)
        anomaly_map[y, x] = mse
        vel_err_map[y, x] = vel_input[i, max_err_index]

        if mse < min_mse and not np.all(spec == 0.0):
            min_mse = mse
            best_index = i
        if mse > max_mse and not np.all(spec == 0.0):
            max_mse = mse
            worst_index = i

    return anomaly_map, vel_err_map, pred_specs, best_index, worst_index


def prepare_input(
    spectra: np.ndarray,
    vels: np.ndarray,
    data: np.ndarray,
    model_hparams: dict,
    model_type: str,
    max_v: float = 5.0,
    max_seq_length: int = 0,
    scale_data: float = -1.0,
) -> (np.ndarray, np.ndarray, dict):
    # # trim spectrum by velocity
    if max_v > 0.0:
        good = np.where(np.abs(vels) <= max_v)
        these_vels = vels[good]
    else:
        these_vels = vels[:]

    if max_seq_length == 0:
        if "autoencoder" in model_type and "max_seq_length" in model_hparams:
            max_seq_length = model_hparams["max_seq_length"]
        elif model_type == "transformer" and "seq_length" in model_hparams:
            max_seq_length = model_hparams["seq_length"]
        elif model_type == "multiloader_transformer":
            max_seq_length = len(these_vels)
        else:
            max_seq_length = 101

    # turn data cubes into model inputs
    num_spectra = len(spectra)
    vel_input = np.zeros((num_spectra, max_seq_length), dtype=np.float32)
    spectra_input = np.zeros((num_spectra, max_seq_length), dtype=np.float32)
    keys = {}

    index = 0

    for y in range(data.shape[1]):
        for x in range(data.shape[2]):
            spec_ = spectra[index]
            spec = spec_[good] if max_v > 0 else spec_[:]
            # print(spec.shape, vels.shape, these_vels.shape)

            # put into zero padded arrays
            if "multiloader" not in model_type:
                filled_spec, filled_vels = fill_spectrum(
                    spec,
                    these_vels,
                    max_seq_length=max_seq_length,
                    scale_data=scale_data,
                )
            elif "autoencoder" in model_type:
                pad_value = model_hparams["pad_value"]
                center = model_hparams["center"]
                random_cycle = model_hparams["random_cycle"]
                if center or random_cycle:
                    filled_vels, filled_spec = cycle_spectrum(
                        these_vels, spec, random_cycle=random_cycle
                    )
                else:
                    filled_vels, filled_spec = these_vels, spec
                if pad_value is not None:
                    spec_len = filled_spec.shape[0]
                    temp_vels = np.full((max_seq_length), pad_value, dtype=np.float32)
                    temp_spectra = np.full((max_seq_length), pad_value, dtype=np.float32)
                    end = spec_len if spec_len <= max_seq_length else max_seq_length
                    temp_vels[:end] = filled_vels[:end]
                    temp_spectra[:end] = filled_spec[:end]
                    filled_vels = temp_vels
                    filled_spec = temp_spectra
            else:
                if scale_data > 0.0:
                    spec *= scale_data
                else:
                    spec = normalize_spectrum(spec)
                    spec *= abs(scale_data)

                filled_spec, filled_vels = spec[:], these_vels[:]

            spectra_input[index, :], vel_input[index, :] = filled_spec, filled_vels

            keys[index] = (x, y)
            index += 1

    return spectra_input, vel_input, keys


def get_anomaly_data(
    fits_name: str,
    test_path: str,
    model_hparams: dict,
    sub_cont: bool = True,
) -> (np.ndarray, np.ndarray, np.ndarray):
    # open fits file
    hdu = fits.open(test_path + fits_name)

    # check if there's a line index (i.e. from MCFOST)
    if "line_index" in model_hparams:
        line_index = model_hparams["line_index"] if "Observations" not in test_path else 0
    elif line_index is None or line_index < 0:
        line_index = 1

    chan_limits = channel_limits[fits_name] if fits_name in channel_limits else (None, None)

    systemic_channel = systemic_channels[fits_name] if fits_name in systemic_channels else 0

    # get line profiles
    vels, spectra = turn_data_cube_into_spectra(
        hdu,
        line_index=line_index,
        sub_cont=sub_cont,
        channel_lims=chan_limits,
        systemic_channel=systemic_channel,
    )

    # initialize map
    data = hdu[0].data.squeeze()
    data[np.isnan(data)] = 0
    if len(data.shape) == 4:
        data = data[line_index, :, :, :]

    if fits_name in channel_limits:
        data = data[chan_limits[0] : chan_limits[1], :, :]

    return vels, spectra, data, hdu


def make_anomaly_map(
    test_path: str = "./Data/Non_Keplerian/Planets/",
    fits_name: str = "13CO_lines_run_0_planet0_00226.fits",
    max_seq_length: int = 0,
    scale_data: float | None = None,
    crop: tuple = ((None, None), (None, None)),
    save: bool = False,
    model_name: str = "upbeat-yogurt-40",
    model_path: str = "./trained_models/",
    use_checkpoint: bool = False,
    #  overlay: bool = False,
    sub_cont: bool = True,
    accelerator_name: str = "cpu",
    save_dir: str = "./Results/",
    max_v: float = 0.0,
    return_data: bool = False,
    inference_batches: int = 0,
    plot_examples: bool = True,
    parameter_path: str = "./Parameters/",
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Makes map of MSE of reconstruction of each pixel in a fits file"""
    print(f"Making map for {fits_name} using model {model_name}")

    # get PyTorch device
    if accelerator_name == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif accelerator_name == "cuda:0":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Loading {model_name}")
    # load model with that name saved at that path
    model = load_trained_model(
        model_name,
        model_path=model_path,
        use_checkpoint=use_checkpoint,
        device=device,
        parameter_path=parameter_path,
    )

    if model_name in all_model_hparams:
        # get the hyperparameters of the model
        model_hparams = all_model_hparams[model_name]
        # the architecture of the model
        model_type = model_types[model_name]
    else:
        model_hparams = yaml.safe_load(Path(f"{parameter_path}{model_name}.yaml").read_text())
        model_type = model_hparams["model_type"]

    # don't do maximum velocity for observations (TODO: change that)
    if "Observations" in test_path:
        max_v = 0.0
    elif "max_v" in model_hparams and max_v == 0.0:
        max_v = model_hparams["max_v"]
    # elif max_v == 0.0:
    # max_v = 5.0

    # determine how to scale data
    if "scale_data" in model_hparams:
        scale_data = model_hparams["scale_data"]
    # default is normalize
    elif scale_data is None:
        scale_data = -1.0

    print(f"Preparing data for {test_path + fits_name}")

    vels, spectra, data, hdu = get_anomaly_data(
        fits_name,
        test_path,
        model_hparams,
        sub_cont=sub_cont,
    )

    spectra_input, vel_input, keys = prepare_input(
        spectra,
        vels,
        data,
        model_hparams,
        model_type,
        max_v=max_v,
        max_seq_length=max_seq_length,
        scale_data=scale_data,
    )

    anomaly_map, vel_err_map, pred_specs, best_index, worst_index = anomaly_inference(
        model,
        spectra_input,
        vel_input,
        keys,
        data,
        device,
        inference_batches=inference_batches,
        pad_value=model_hparams.get("pad_value", None),
    )

    print("Plotting map")

    mplrc("xtick", labelsize=1.0 * ticks)
    mplrc("ytick", labelsize=1.0 * ticks)

    fig = plt.figure(figsize=(12.5, 10.0))

    temp_hdu = hdu
    # set middle to 0 in order to just get angular size (don't care about position)
    temp_hdu[0].header["CRVAL1"] = 0.0
    temp_hdu[0].header["CRVAL2"] = 0.0

    # put RA and DEC on
    wcs = WCS(temp_hdu[0].header, naxis=2)

    ax = WCSAxes(
        fig,
        [0.1, 0.1, 0.8, 0.8],
        # slices=('x', 'y', 0, 0),
        wcs=wcs,
    )
    fig.add_axes(ax)

    RA = ax.coords[0]
    DEC = ax.coords[1]

    RA.set_ticks(number=5, exclude_overlapping=True)
    DEC.set_ticks(number=5, exclude_overlapping=True)

    plt.imshow(anomaly_map, cmap="Reds", origin="lower", norm=LogNorm())
    # plt.contour(np.sum(data, axis=0), color="white")
    cbar = plt.colorbar(fraction=0.048, pad=0.005)  # , norm=LogNorm())
    cbar.ax.set_ylabel("MSE", rotation=270, fontsize=titles, labelpad=30)
    cbar.ax.tick_params(labelsize=ticks)

    plt.xticks(fontsize=ticks)
    plt.yticks(fontsize=ticks)

    plt.xlabel(r"$\Delta$ RA", fontsize=labels)
    plt.ylabel(r"$\Delta$ DEC", fontsize=labels)

    if fits_name in translations:
        plt.title(translations[fits_name], fontsize=titles)

    if crop[0][0] is not None:
        center_x = hdu[0].header["CRPIX1"]
        plt.xlim(center_x - crop[0][0], center_x + crop[0][1])
    if crop[1][0] is not None:
        center_y = hdu[0].header["CRPIX2"]
        plt.ylim(center_y - crop[1][0], center_y + crop[1][1])

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if not os.path.exists(f"{save_dir}Plots/"):
            os.mkdir(f"{save_dir}Plots/")

        plt.savefig(
            f"{save_dir}Plots/anomaly_map_{fits_name}_{model_name}_sub_cont_{sub_cont}.pdf"
        )

    plt.show(block=False)
    plt.pause(10)
    plt.close()

    mplrc("xtick", labelsize=1.0 * ticks)
    mplrc("ytick", labelsize=1.0 * ticks)

    fig = plt.figure(figsize=(12.5, 10.0))

    temp_hdu = hdu
    # set middle to 0 in order to just get angular size (don't care about position)
    temp_hdu[0].header["CRVAL1"] = 0.0
    temp_hdu[0].header["CRVAL2"] = 0.0

    # put RA and DEC on
    wcs = WCS(temp_hdu[0].header, naxis=2)

    ax = WCSAxes(
        fig,
        [0.1, 0.1, 0.8, 0.8],
        # slices=('x', 'y', 0, 0),
        wcs=wcs,
    )
    fig.add_axes(ax)

    RA = ax.coords[0]
    DEC = ax.coords[1]

    RA.set_ticks(number=5, exclude_overlapping=True)
    DEC.set_ticks(number=5, exclude_overlapping=True)

    plt.imshow(vel_err_map, cmap="RdBu", origin="lower")
    # plt.contour(np.sum(data, axis=0), color="white")
    cbar = plt.colorbar(fraction=0.048, pad=0.005)  # , norm=LogNorm())
    cbar.ax.set_ylabel("Velocity [km/s]", rotation=270, fontsize=titles, labelpad=30)
    cbar.ax.tick_params(labelsize=ticks)

    plt.xticks(fontsize=ticks)
    plt.yticks(fontsize=ticks)

    plt.xlabel(r"$\Delta$ RA", fontsize=labels)
    plt.ylabel(r"$\Delta$ DEC", fontsize=labels)

    if fits_name in translations:
        plt.title(translations[fits_name], fontsize=titles)

    if crop[0][0] is not None:
        center_x = hdu[0].header["CRPIX1"]
        plt.xlim(center_x - crop[0][0], center_x + crop[0][1])
    if crop[1][0] is not None:
        center_y = hdu[0].header["CRPIX2"]
        plt.ylim(center_y - crop[1][0], center_y + crop[1][1])

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if not os.path.exists(f"{save_dir}Plots/"):
            os.mkdir(f"{save_dir}Plots/")

        plt.savefig(
            f"{save_dir}Plots/max_v_err_map_{fits_name}_{model_name}_sub_cont_{sub_cont}.pdf"
        )

    plt.show(block=False)
    plt.pause(10)
    plt.close()

    if plot_examples:
        # plot the best, worst, and random spectrum
        example_indices = [best_index, worst_index]
        i_ = np.random.randint(low=0, high=len(spectra_input))
        while i_ in example_indices or np.all(spectra_input[i_] == 0.0):
            i_ = np.random.randint(low=0, high=len(spectra_input))
        example_indices.append(i_)

        for i in example_indices:
            fig = plt.figure(figsize=(14.5, 10.0))

            pad_value = model_hparams.get("pad_value", None)
            if pad_value is not None:
                good = np.where(spectra_input[i] != pad_value)
                spectrum = spectra_input[i, good].squeeze()
                pred_spec = pred_specs[i, good].squeeze()
            else:
                spectrum = spectra_input[i, :]
                pred_spec = pred_specs[i, :]

            plt.plot(
                range(len(spectrum)),
                spectrum,
                lw=lw,
                c="firebrick",
                label="Actual",
            )
            plt.plot(
                range(len(spectrum)),
                pred_spec,
                lw=lw,
                c="steelblue",
                label="Predicted",
            )

            plt.xticks(fontsize=ticks)
            plt.yticks(fontsize=ticks)

            plt.xlabel(r"Index", fontsize=labels)
            plt.ylabel(r"Intensity (normalized)", fontsize=labels)

            plt.legend(loc="best", fontsize=legends)

            i_str = f"{i}"
            if i != i_:
                i_str = f"best_{i}" if i == best_index else f"worst_{i}"

            plt.savefig(
                f"{save_dir}Plots/pred_real_spec_{i_str}_{fits_name}_{model_name}_sub_cont_{sub_cont}.pdf"
            )

            plt.show(block=False)
            plt.pause(5)
            plt.close()

    print("")

    if return_data:
        return pred_specs, spectra_input, vel_input, map
    return None


def get_fits_in_dir(test_path: str = "./Data/") -> list:
    """Gets the directory and name of every fits file in a directory"""
    ### TODO: not working for mutliple directories
    all_files = os.listdir(test_path)
    test_paths = []
    fits_names = []
    for file in all_files:
        print(file)
        if "Continuum" in file or "data_th" in file or ".gz" in file:
            continue
        if ".fits" in file:
            test_paths.append(test_path)
            fits_names.append(file)
        elif os.path.isdir(f"{test_path}{file}/"):
            new_files, new_paths = get_fits_in_dir(f"{test_path}{file}/")
            if len(new_files) > 0:
                test_paths.extend(new_paths)
                fits_names.extend(new_files)

    return fits_names, test_paths
