import os

import numpy as np
import pytorch_lightning as pl
import torch
from astropy.io import fits
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# sys.path.insert(0, "./")
from utils.globals import c_kms


def get_vels_from_freq(hdr, relative: bool = True, syst_chan: int = 0) -> np.ndarray:
    """Doppler shift formula to get velocity"""
    f0 = hdr["CRVAL3"]
    delta_f = hdr["CDELT3"]
    center = int(hdr["CRPIX3"])
    num_freq = int(hdr["NAXIS3"])
    freqs = [f0 + delta_f * (i - center) for i in range(center - 1, num_freq)]
    vels = np.array([-c_kms * (f - f0) / f0 for f in freqs])
    if relative:
        if syst_chan == 0:
            syst_chan = num_freq // 2
        vels -= vels[syst_chan]

    return vels


def get_vels_from_dv(hdu: list) -> np.ndarray:
    """Gets velocity axis from a fits header in km/s (MCFOST output)"""
    vels = []
    for i in range(hdu[0].header["NAXIS3"]):
        vel = (
            hdu[0].header["CDELT3"] * (i + 1 - hdu[0].header["CRPIX3"]) + hdu[0].header["CRVAL3"]
        )
        vels.append(vel)

    return np.array(vels)


def load_fits_file(path: str) -> list:
    return fits.open(path)


def get_all_data_paths(directory: str, one_per_run: bool = False) -> list:
    """Gets the path and label of each fits file in the directory"""
    data = os.listdir(directory)

    paths = [f"{directory}{x}" for x in data if ".fits" in x]

    # make sure there are no duplicates from same simulation
    if one_per_run:
        runs = []
        for path in paths:
            this_path = path.split("_")
            i = 0
            while this_path[i] != "run":
                i += 1
            i += 1
            runs.append(this_path[i])
        temp_paths = []
        used_runs = []
        for i, path in enumerate(paths):
            this_run = runs[i]
            # don't include any with the same run
            if this_run in used_runs:
                continue
            temp_paths.append(path)
            used_runs.append(this_run)
        paths = temp_paths

    return paths


def get_dirty_paths(
    directory: str,
    one_per_run: bool = False,
    dirty_ext: str = "Dirty/",
) -> (list, list):
    """Get paths to both the raw and CASA processed fits files.
    Ensures that they are corresponding by checking run_XXX and step_XXXX"""
    clean_paths = get_all_data_paths(directory)
    dirty_dir = directory + dirty_ext
    dirty_paths = get_all_data_paths(dirty_dir)

    if not one_per_run:
        return clean_paths, dirty_paths

    clean_runs = []
    runs = []
    steps = []
    # get paths to raw fits files
    for path in clean_paths:
        this_path = path.split("_")
        i = 0
        # this finds the run name
        while this_path[i] != "run":
            i += 1
        i += 1
        clean_runs.append(this_path[i])
        # this finds the step
        while "planet" not in this_path[i]:
            i += 1
        i += 1
        steps.append(this_path[i])
    # get rid of any duplicates from the same simulation
    temp_paths = []
    used_runs = []
    used_steps = []
    for i, path in enumerate(clean_paths):
        this_run = runs[i]
        this_step = steps[i]
        if this_run in used_runs:
            continue
        temp_paths.append(path)
        used_runs.append(this_run)
        used_steps.append(this_step)
    clean_paths = temp_paths

    used_dirty_runs = []
    temp_dirty_paths = []
    # get the corresponding CASA paths
    for path in dirty_paths:
        this_path = path.split("_")
        i = 0
        while this_path[i] != "run":
            i += 1
        i += 1
        this_run = path[i]
        while "planet" not in this_path[i]:
            i += 1
        i += 1
        this_step = this_path[i]

        if this_run in used_dirty_runs:
            continue

        for i, clean_run in enumerate(clean_runs):
            clean_step = used_steps[i]
            if this_step == clean_step and this_run == clean_run:
                used_dirty_runs.append(this_run)
                temp_dirty_paths.append(path)
                break

    dirty_paths = temp_dirty_paths

    return clean_paths, dirty_paths


def load_all_data_cubes(
    directory: str, one_per_run: bool = False, dirty: bool = False, dirty_ext: str = "Dirty/"
) -> tuple[dict, (dict, dict)]:
    """Gets the paths to all fits files in the directories"""
    if not dirty:
        paths = get_all_data_paths(directory, one_per_run=one_per_run)
        if not one_per_run:
            paths = paths[::2]

        return {path: load_fits_file(path) for path in paths}

    clean_paths, dirty_paths = get_dirty_paths(
        directory, one_per_run=one_per_run, dirty_ext=dirty_ext
    )

    return {path: load_fits_file(path) for path in clean_paths}, {
        path: load_fits_file(dirty_paths[i]) for i, path in enumerate(clean_paths)
    }


def turn_data_cube_into_spectra(
    hdu: list,
    line_index: int = 1,
    sub_cont: bool = False,
    channel_lims: tuple = (None, None),
    systemic_channel: int = 0,
) -> (np.ndarray, np.ndarray):
    """Take a fits file and turn it into line profiles"""
    # access HDU data and get rid of any dimensions of 1
    data = hdu[0].data.squeeze()
    # get rid of NaNs
    data[np.isnan(data)] = 0
    # MCFOST data has a line index axis
    if len(data.shape) == 4:
        data = data[line_index, :, :, :]

    if channel_lims[0] is not None and channel_lims[1] is not None:
        data = data[channel_lims[0] : channel_lims[1], :, :]

    # subtract continuum (highest and lowest velocity channels)
    if sub_cont:
        data -= data[0, :, :]
        data -= data[-1, :, :]
        data[data < 0.0] = 0.0

    num_spectra = data.shape[1] * data.shape[2]

    # get velocities from fits headers
    num_vels = data.shape[0]
    if hdu[0].header["CTYPE3"] == "VELO-LSR":
        vels = get_vels_from_dv(hdu).astype(np.float32)
    elif hdu[0].header["CTYPE3"] == "FREQ":
        vels = get_vels_from_freq(hdu[0].header, syst_chan=systemic_channel).astype(np.float32)

    if channel_lims[0] is not None and channel_lims[1] is not None:
        vels = vels[channel_lims[0] : channel_lims[1]]

    spectra = np.empty((num_spectra, num_vels), dtype=np.float32)

    # get line profiles
    num_used_spectra = 0
    for y in range(data.shape[1]):
        for x in range(data.shape[2]):
            spectra[num_used_spectra, :] = data[:, y, x]
            num_used_spectra += 1

    return vels, spectra


def get_all_spectra(
    directory: str,
    line_index: int = 1,
    one_per_run: bool = False,
    dirty: bool = False,
    dirty_ext: str = "Dirty/",
) -> tuple[(dict, dict), (dict, dict, dict, dict)]:
    """Loads all fits files and turns them into line profiles"""
    # load all data cubes
    cubes = load_all_data_cubes(
        directory, one_per_run=one_per_run, dirty=dirty, dirty_ext=dirty_ext
    )

    # if not doing domain adaptation
    if not dirty:
        vels, spectra = {}, {}
        # turn each cube into line profile
        for run in cubes:
            vels[run], spectra[run] = turn_data_cube_into_spectra(
                cubes[run], line_index=line_index
            )

        return vels, spectra

    # do the same for both the clean and dirty spectra if doing domain adaptation
    clean_vels, clean_spectra, dirty_vels, dirty_spectra = {}, {}, {}, {}
    clean_cubes, dirty_cubes = cubes[:]

    for run in clean_cubes:
        clean_vels[run], clean_spectra[run] = turn_data_cube_into_spectra(
            clean_cubes[run], line_index=line_index
        )
        dirty_vels[run], dirty_spectra[run] = turn_data_cube_into_spectra(
            dirty_cubes[run], line_index=line_index
        )

        return clean_spectra, dirty_spectra, clean_vels, clean_spectra
    return None


def fill_spectrum(
    spec: np.ndarray, these_vels: np.ndarray, max_seq_length: int = 101, scale_data: float = -1.0
) -> (np.ndarray, np.ndarray):
    """Fills a zero-padded array of a given length with a spectrum of possibly different length.
    If the spectrum is too long, it is encodded to use the maximum intensity is included.
    If the spectrum is too short, it is put in the middle of the padded array.
    """
    # initialize arrays
    empty_spectrum = np.zeros((max_seq_length), dtype=np.float32)
    empty_vels = np.zeros((max_seq_length), dtype=np.float32)
    length = len(spec)

    # find where the maximum intensity is
    peak_index = np.where(spec == np.max(spec))
    while type(peak_index) in (tuple, np.ndarray, list):
        peak_index = peak_index[0]

    if np.all(spec == 0):
        peak_index = length // 2

    # simply fill in if they are the same size
    if length == max_seq_length:
        empty_spectrum[:] = spec[:]
        empty_vels[:] = these_vels[:]
    # fill in spectrum in center of padded array if it's shorter
    elif length < max_seq_length:
        width = length // 2
        empty_spectrum[
            max_seq_length // 2 - width : max_seq_length // 2 + width + int(length % 2 == 1)
        ] = spec[:]
        empty_vels[
            max_seq_length // 2 - width : max_seq_length // 2 + width + int(length % 2 == 1)
        ] = these_vels[:]
        assert np.max(spec) in empty_spectrum, "Maximum not in data"
    # ensure the maximum is included if the spectrum needs to be trimmed
    else:
        width = max_seq_length // 2
        # if it's in the second half of the spectrum
        if peak_index >= width:
            # take into account it it's too close to the end
            if peak_index + width + int(max_seq_length % 2 == 1) > length:
                offset = length - (peak_index + width + int(max_seq_length % 2 == 1))
            else:
                offset = 0
            # trim spectrum around maximum
            spec = spec[
                peak_index
                - width
                + offset : peak_index
                + width
                + offset
                + int(max_seq_length % 2 == 1)
            ]
            these_vels = these_vels[
                peak_index
                - width
                + offset : peak_index
                + width
                + offset
                + int(max_seq_length % 2 == 1)
            ]
            # fill in padded spectrum
            empty_spectrum[: len(spec)] = spec
            empty_vels[: len(these_vels)] = these_vels
        else:
            # if it's in the first half of the data, just feed that in
            empty_spectrum[:] = spec[:max_seq_length]
            empty_vels[:] = these_vels[:max_seq_length]
        assert np.max(spec) in empty_spectrum, "Maximum not in data"

    # -1 means normalize
    if scale_data == -1.0 and not np.all(empty_spectrum == 0.0):
        normalized = empty_spectrum.copy() - np.min(empty_spectrum.copy())
        if np.max(normalized) != 0.0:
            normalized /= np.max(normalized)
            filled_spec = normalized
            assert (
                np.max(filled_spec) <= 1.0 and np.min(filled_spec) >= 0.0
            ), "Input improperly normalized"
        else:
            filled_spec = empty_spectrum
    else:
        filled_spec = scale_data * empty_spectrum

    filled_vels = empty_vels[:]

    return filled_spec.astype(np.float32), filled_vels.astype(np.float32)


def make_all_data(
    directory: str,
    line_index: int = 1,
    max_v: float = 5.0,
    max_seq_length: int = 100,
    scale_data: float = 1.0,
    one_per_run: bool = False,
    dirty: bool = False,
    dirty_ext: str = "Dirty/",
    ignore_zeros: bool = False,
) -> tuple[(np.ndarray, np.ndarray), (np.ndarray, np.ndarray, np.ndarray, np.ndarray)]:
    """Gets all spectra (and CASA corrupted for dirty=True) within a directory"""
    outputs = get_all_spectra(
        directory,
        line_index=line_index,
        one_per_run=one_per_run,
        dirty=dirty,
        dirty_ext=dirty_ext,
    )

    if not dirty:
        vels, spectra = outputs[:]
        order = list(spectra.keys())
        return get_spectra_vel_input(
            spectra,
            vels,
            order,
            max_seq_length=max_seq_length,
            scale_data=scale_data,
            max_v=max_v,
            ignore_zeros=ignore_zeros,
        )

    clean_spectra, dirty_spectra, clean_vels, dirty_vels = outputs[:]
    order = list(spectra.keys)
    clean_input_spec, clean_input_vels = get_spectra_vel_input(
        clean_spectra,
        clean_vels,
        order,
        max_seq_length=max_seq_length,
        scale_data=scale_data,
        max_v=max_v,
        ignore_zeros=ignore_zeros,
    )
    dirty_input_spec, dirty_input_vels = get_spectra_vel_input(
        dirty_spectra,
        dirty_vels,
        order,
        max_seq_length=max_seq_length,
        scale_data=scale_data,
        max_v=max_v,
        ignore_zeros=ignore_zeros,
    )

    return clean_input_spec, clean_input_vels, dirty_input_spec, dirty_input_vels


def get_spectra_vel_input(
    spectra: dict,
    vels: dict,
    order: list,
    max_seq_length: int = 101,
    scale_data: float = 1.0,
    max_v: float = 5.0,
    ignore_zeros: bool = False,
):
    num_spectra = sum(spectra[run].shape[0] for run in order)
    vel_input = np.zeros((num_spectra, max_seq_length), dtype=np.float32)
    spectra_input = np.zeros((num_spectra, max_seq_length), dtype=np.float32)

    index = 0
    for run in order:
        print(f"Getting data from {run}")
        these_vels = vels[run]
        this_spectrum = spectra[run]

        if max_v > 0:
            good = np.where(np.abs(these_vels) <= max_v)
            these_vels = these_vels[good]
            # this_spectrum = this_spectrum[good]
        for spec_ in this_spectrum:
            if ignore_zeros and np.all(spec_ == 0.0):
                continue

            spec = spec_[good] if max_v > 0.0 else spec_[:]

            filled_spec, filled_vels = fill_spectrum(
                spec,
                these_vels,
                max_seq_length=max_seq_length,
                scale_data=scale_data,
            )
            spectra_input[index, :] = filled_spec
            vel_input[index, :] = filled_vels

            index += 1

    # return spectra_input.astype(np.float32), vel_input.astype(np.float32)

    # takes into account spectra with only zeros
    if ignore_zeros and index < num_spectra:
        spectra_input, vel_input = spectra_input[:index, :], vel_input[:index, :]

    return spectra_input, vel_input


def prepare_datasets(
    data_path: str = "./data/",
    scale_data: float = 1.0,
    max_seq_length: int = 100,
    val_split: float = 0.2,
    test_split: float = 0.25,
    line_index: int = 1,
    max_v: float = 5.0,
    one_per_run: bool = False,
    dirty: bool = False,
    dirty_ext: bool = False,
    encoder_model: tuple[pl.LightningModule, None] | None = None,
    accelerator_name: str = "mps",
    ignore_zeros: bool = False,
) -> (Dataset, Dataset, Dataset):
    """Gets all spectra and velocities within a given data directory
    dirty is for domain adaptation and also loads corresponding CASA-corrupted files
    """
    if not dirty:
        spectra, velocities = make_all_data(
            data_path,
            scale_data=scale_data,
            line_index=line_index,
            max_v=max_v,
            max_seq_length=max_seq_length,
            one_per_run=one_per_run,
            ignore_zeros=ignore_zeros,
        )

        train_spec, test_spec, train_vels, test_vels = train_test_split(
            spectra, velocities, test_size=test_split, random_state=123
        )

        train_spec, val_spec, train_vels, val_vels = train_test_split(
            train_spec, train_vels, test_size=val_split, random_state=123
        )

        train_set = SpectrumDataset(train_vels, train_spec, accelerator_name=accelerator_name)
        val_set = SpectrumDataset(val_vels, val_spec, accelerator_name=accelerator_name)
        test_set = SpectrumDataset(test_vels, test_spec, accelerator_name=accelerator_name)

        return train_set, val_set, test_set

    # get the clean and CASA corrupted spectra
    clean_spectra, _, dirty_spectra, _ = make_all_data(
        data_path,
        scale_data=scale_data,
        line_index=line_index,
        max_v=max_v,
        max_seq_length=max_seq_length,
        one_per_run=one_per_run,
        dirty=dirty,
        dirty_ext=dirty_ext,
        ignore_zeros=ignore_zeros,
    )

    ### just going to ignore velocities
    # split into train, test, validation
    train_clean, test_clean, train_dirty, test_dirty = train_test_split(
        clean_spectra, dirty_spectra, test_size=test_split, random_state=123
    )

    train_clean, val_clean, train_dirty, val_dirty = train_test_split(
        train_clean, train_dirty, test_size=val_split, random_state=123
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # encode the clean spectra
    train_clean = torch.from_numpy(train_clean).float().to(device)
    test_clean = torch.from_numpy(test_clean).float().to(device)
    val_clean = torch.from_numpy(val_clean).float().to(device)

    train_clean_enc = encoder_model(train_clean).detach().cpu().numpy()
    test_clean_enc = encoder_model(test_clean).detach().cpu().numpy()
    val_clean_enc = encoder_model(val_clean).detach().cpu().numpy()

    # make datasets
    train_set = SpectrumDataset(train_clean_enc, train_dirty)
    val_set = SpectrumDataset(val_clean_enc, val_dirty)
    test_set = SpectrumDataset(test_clean_enc, test_dirty)

    return train_set, val_set, test_set


class SpectrumDataset(Dataset):

    """Data loader"""

    def __init__(
        self,
        velocities,
        intensities,
        convert_y_to_float: bool = True,
        accelerator_name: str = "mps",
    ) -> None:
        self.velocities, self.intensities = velocities, intensities
        self.convert_y_to_float = convert_y_to_float

        if accelerator_name == "mps":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif accelerator_name == "cuda:0":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return len(self.velocities)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        x_, y_ = self.velocities[idx], self.intensities[idx]
        # return x_, y_
        # return torch.tensor(x_, dtype=torch.double), y_
        # print("Dataset sizes ", x_.shape, omegas_.shape, y_.shape)
        if not self.convert_y_to_float:
            return torch.from_numpy(x_).float().to(self.device), y_
        return (
            torch.from_numpy(x_).float().to(self.device),
            torch.from_numpy(y_).float().to(self.device),
        )
