"""
One protocol, every checkpoint.

Earlier, each checkpoint family had its own scoring script with its own preprocessing, so no two checkpoints had
been put through the same checks on the same cubes. This module gives every checkpoint one interface and runs
the same battery on the same cases. It only composes functions that already exist and have published numbers
(`moment_improvement`, `quadratic_moment1`, `fit_keplerian`, `channel_artifacts`, `plot_moment_comparison`), so
a score here is comparable with the standing tables.

The checks, per (checkpoint, case):

    pixel       PSNR / SSIM, per channel on the shared dirty-scale normalisation, clamped to [0, 1]
    moments     M0 / M1 / M2 improvement over dirty, clipped + signal-masked (`moment_improvement`)
    wiggle      Keplerian-residual correlation with clean, against ONE shared geometry per case
    sharpness   gradient energy and Laplacian variance of the raw quadratic M1, as a ratio to clean's
    artifacts   invented-structure rate, blob count, overshoot, floor leak (`channel_artifacts`)
    figure      moment-map comparison panel (optional, it is plotting, not compute)
    transfer    no separate code: a line-emission checkpoint scored on an SG case (and the reverse) is
                the same call with a different case, so the cross-domain matrix falls out of the loop
    headroom    the (dirty resid_r, model gain) pair is in every row, ready for the scatter

Evaluation space. Every case is continuum-subtracted (mean of the first and last `CONTINUUM_N` channels),
which is how notebooks 05 and 08 score. A checkpoint trained without continuum subtraction (the SG family,
notebook 12 `subtract_continuum=False`) is fed the same continuum-subtracted cube; on SG data the continuum is
~0 so this is a no-op there, and on line-emission cubes it is the closest available input. State this
wherever such a cross-domain number is quoted.

Nothing here decides which loss is best. 
"""
from __future__ import annotations

import glob
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

CONTINUUM_N = 5
FRAC = 0.05            # signal-mask fraction, the project's plateau value (moment_maps.SIGNAL_FRAC)
SSIM_STRIDE = 2        # SSIM on every 2nd channel: it is the slow step and adds nothing at full stride

ROW_FIELDS = [
    "checkpoint", "source", "family", "case", "domain", "n_channels",
    "amp_scale", "psnr", "psnr_dirty", "ssim", "ssim_dirty",
    "M0", "M1", "M2", "M0_all", "M1_all", "M2_all", "n_px",
    "resid_r", "dirty_resid_r", "wiggle_gain", "resid_err_ratio", "resid_rms_ratio", "mstar_at_bound", "geom_ok", "geom_offset_px", "geom_mstar", "ref_resid_rms",
    "gradE_ratio", "gradE_ratio_dirty", "lapvar_ratio", "lapvar_ratio_dirty",
    "invented_frac", "invented_blobs", "overshoot", "floor_leak",
    "wall_s", "note",
]


# ------------------------------------------------------------------------------------------------ #
# Checkpoints                                                                                       #
# ------------------------------------------------------------------------------------------------ #
_EXT = (".pth.tar", ".pth", ".ckpt")
_PREFIX = re.compile(r"^(nb\d+_)")


def label_of(path: str) -> str:
    """'nb05_winner_starlet_ft_seed42.pth' -> 'winner_starlet_ft_seed42'. Seed is kept: it is identity."""
    base = os.path.basename(path)
    for e in _EXT:
        if base.endswith(e):
            base = base[: -len(e)]
            break
    return _PREFIX.sub("", base)


def source_of(path: str) -> str:
    m = _PREFIX.match(os.path.basename(path))
    return m.group(1).rstrip("_") if m else "best_models"


def discover(roots: Sequence[str], patterns: Sequence[str] = ("*.pth", "*.ckpt", "*.pth.tar")) -> Dict[str, str]:
    """
    {label: path} for every checkpoint under `roots`. A label found twice keeps the first path, so
    put the authoritative root first. Files that are not torch checkpoints are the caller's problem:
    `describe` raises on them rather than this function guessing from the name.
    """
    found: Dict[str, str] = {}
    for root in roots:
        for pat in patterns:
            for p in sorted(glob.glob(os.path.join(root, "**", pat), recursive=True)):
                if os.path.isfile(p):
                    found.setdefault(label_of(p), p)
    return found


@dataclass
class Spec:
    path: str
    label: str
    source: str
    kind: str                 # "unet" | "diffusion"
    family: str               # unet | stack_kin | stack_sg | stack | ddpm | ddrm
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 48
    channel_multipliers: tuple = (1, 2, 4, 8)
    beam_dim: int = 0
    loss_name: str = ""
    size: int = 256           # pixel grid the model was trained on
    supported: bool = True
    why_not: str = ""

    @property
    def K(self) -> int:
        return (self.in_channels - 1) // 2


def _train_size(label: str) -> int:
    for tag, size in (("native600", 600), ("res480", 480), ("res320", 320)):
        if tag in label:
            return size
    # notebook 14's native-resolution arms are named sweep_winner_600 / _aug_600 / _p10_600, with no "native".
    # Until 2026-09-25 they fell through to 256 and would have been scored at a scale they never saw.
    if re.search(r"(^|_)600($|_)", label):
        return 600
    return 256


def describe(path: str) -> Spec:
    """Read a checkpoint's own metadata. Never guesses architecture from the filename."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    label, source = label_of(path), source_of(path)
    if "model_state_dict" in ck:
        sd = ck["model_state_dict"]
        n_in = int(ck.get("in_channels", 1))
        n_out = int(ck.get("out_channels", 1))
        if n_in == 1 and n_out == 1:
            fam = "unet"
        elif n_in == 7 and n_out == 1:
            fam = "stack_sg"
        elif n_in == n_out and n_in > 1:
            fam = "stack_kin"
        else:
            fam = "stack"
        del sd
        return Spec(path, label, source, "unet", fam, n_in, n_out, int(ck["base_channels"]),
                    tuple(ck["channel_multipliers"]), int(ck.get("beam_dim", 0)),
                    str(ck.get("loss_name", "")), _train_size(label))
    if "state_dict" in ck and "config" in ck:
        cond = bool(ck["config"]["data"]["conditional"])
        size = int(ck["config"]["data"]["image_size"])
        if cond:
            return Spec(path, label, source, "diffusion", "ddpm", 1, 1, size=size,
                        loss_name=str(ck["config"]["diffusion"].get("loss_type", "l2")))
        return Spec(path, label, source, "diffusion", "ddrm", 1, 1, size=size, supported=False,
                    why_not="DDRM restores through the beam operator, which needs a per-cube beam "
                            "transfer function and measurement noise; not wired into this protocol yet")
    raise ValueError(f"{path}: neither a U-Net checkpoint (model_state_dict) nor a diffusion one (state_dict + config)")


class Denoiser:
    """
    One callable per checkpoint: `den = Denoiser(spec, device)`, then `den(dirty_cube, beam_vec)` returns the
    denoised cube in the SAME space and shape as the input. All preprocessing (neighbour stack, per-channel
    min-max shared from the centre channel, resize to the model's grid) lives here so the caller cannot get
    it wrong for one family and right for another.
    """

    def __init__(self, spec: Spec, device: str = "cuda", sampling_steps: int = 25, n_avg: int = 1):
        import torch
        self.spec, self.torch = spec, torch
        self.device = torch.device(device)
        self.sampling_steps, self.n_avg = sampling_steps, n_avg
        if not spec.supported:
            raise NotImplementedError(f"{spec.label}: {spec.why_not}")
        if spec.kind == "unet":
            from src.models.unet import UNet
            ck = torch.load(spec.path, map_location=self.device, weights_only=False)
            self.net = UNet(in_channels=spec.in_channels, out_channels=spec.out_channels,
                            base_channels=spec.base_channels, channel_multipliers=spec.channel_multipliers,
                            time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, spec.base_channels),
                            beam_dim=spec.beam_dim).to(self.device)
            self.net.load_state_dict(ck["model_state_dict"], strict=True)
            self.net.eval()
        else:
            from src.training.diffusion import DenoisingDiffusion
            ck = torch.load(spec.path, map_location=self.device, weights_only=False)
            self.dd = DenoisingDiffusion(config=ck["config"], device=str(self.device), lr=1e-4,
                                         checkpoint_path=spec.path, data_parallel=False)
            self.dd.load_checkpoint(spec.path)
            self.dd.model.eval()

    # -- U-Net (single channel, neighbour stack, beam-conditioned) ------------------------------ #
    def _run_unet(self, dirty, beam_vec, batch):
        import torch
        import torch.nn.functional as F
        from src.training.architectures import forward_fn
        fwd = forward_fn("unet")
        sp = self.spec
        K, S = sp.K, sp.size
        out_idx = K if sp.out_channels > 1 else 0
        C, H, W = dirty.shape
        out = np.empty((C, H, W), dtype=np.float32)
        beam = None
        if sp.beam_dim:
            bv = np.zeros(4, np.float32) if beam_vec is None else np.asarray(beam_vec, np.float32)
            beam_t = torch.from_numpy(bv)[None].to(self.device)
        with torch.no_grad():
            for s in range(0, C, batch):
                cen = np.arange(s, min(s + batch, C))
                nb = np.clip(cen[:, None] + np.arange(-K, K + 1)[None, :], 0, C - 1)
                stack = dirty[nb].astype(np.float32)                      # (b, 2K+1, H, W)
                ref = dirty[cen]
                lo = ref.reshape(len(cen), -1).min(axis=1)
                hi = ref.reshape(len(cen), -1).max(axis=1)
                rng = np.where(hi > lo, hi - lo, 1.0)
                n = (stack - lo[:, None, None, None]) / rng[:, None, None, None]   # centre-channel scale for the WHOLE stack
                t = torch.from_numpy(n).to(self.device)
                t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)
                if sp.beam_dim:
                    beam = beam_t.expand(len(cen), -1)
                pred, _ = fwd(self.net, t, beam)
                pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
                centre = pred[:, out_idx].cpu().numpy()
                for j, c in enumerate(cen):
                    out[c] = centre[j] * rng[j] + lo[j] if hi[j] > lo[j] else lo[j]
        return out

    # -- conditional DDPM ------------------------------------------------------------------------ #
    def _run_ddpm(self, dirty, batch):
        import torch
        import torch.nn.functional as F
        S = self.spec.size
        C, H, W = dirty.shape
        out = np.empty((C, H, W), dtype=np.float32)
        for s in range(0, C, batch):
            cen = np.arange(s, min(s + batch, C))
            ref = dirty[cen]
            lo = ref.reshape(len(cen), -1).min(axis=1)
            hi = ref.reshape(len(cen), -1).max(axis=1)
            rng = np.where(hi > lo, hi - lo, 1.0)
            n = (ref - lo[:, None, None]) / rng[:, None, None]
            t = torch.from_numpy(n)[:, None].float().to(self.device)
            t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)
            with torch.no_grad():
                p = self.dd.sample(t, sampling_timesteps=self.sampling_steps, use_ema=True, n_avg=self.n_avg)
                p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for j, c in enumerate(cen):
                out[c] = p[j] * rng[j] + lo[j] if hi[j] > lo[j] else lo[j]
        return out

    def __call__(self, dirty: np.ndarray, beam_vec=None, batch: int = 8) -> np.ndarray:
        if self.spec.kind == "diffusion":
            return self._run_ddpm(dirty, batch=max(1, batch // 4))
        return self._run_unet(dirty, beam_vec, batch)


# ------------------------------------------------------------------------------------------------ #
# Cases                                                                                             #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Case:
    name: str
    domain: str                      # "line_emission" | "sg"
    clean: np.ndarray                # (C, H, W) continuum-subtracted
    dirty: np.ndarray
    velax: np.ndarray                # m/s
    au_per_px: float
    beam_vec: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    fixed_incl: Optional[float] = None
    meta: dict = field(default_factory=dict)


def _hdr_and_data(path):
    from astropy.io import fits
    h = fits.open(path, memmap=True)
    return h, h[0].header, h[0].data


def _csub(cube_full, n=CONTINUUM_N):
    from src.data.fits_cube_dataset import continuum_of
    return continuum_of(cube_full, n)


def _edge_mean(cube, n=CONTINUUM_N) -> np.ndarray:
    """`continuum_of` for a memmapped cube: mean of the first and last n planes, read without loading the rest."""
    return np.concatenate([np.asarray(cube[:n], np.float32), np.asarray(cube[-n:], np.float32)]).mean(axis=0)


def _para_value(folder: str, needle: str) -> Optional[float]:
    for p in glob.glob(os.path.join(folder, "*.para")):
        for line in open(p):
            if needle in line:
                try:
                    return float(line.split()[0])
                except ValueError:
                    return None
    return None


def line_emission_cases(data_dir: str, limit: Optional[int] = None, channels: Optional[slice] = None,
                        fix_incl: bool = True) -> Iterable[Case]:
    """
    The 5 held-out line-emission cubes (RunIDs 0002 x3, 0025, 0026), the set notebooks 05 and 08 score on.

    `fix_incl`: hold the Keplerian fit's inclination at the true value from the cube's `.para` file. That is what
    `fit_keplerian`'s own docstring prescribes whenever inclination is known independently, because mass and
    inclination are nearly degenerate. The free fit was degenerate on two of these
    five cubes; False reproduces it.
    """
    from src.data.cube_split import split_cubes
    from src.data.fits_cube_dataset import beam_features_of
    _, _, holdout = split_cubes(data_dir=data_dir, n_holdout=3, val_fraction=0.2, seed=42)
    for ho in holdout[:limit]:
        hc, hdr, cdata = _hdr_and_data(ho["clean"])
        hd, _, ddata = _hdr_and_data(ho["dirty"])
        clean_full = np.asarray(cdata, np.float32)
        dirty_full = np.asarray(ddata, np.float32)
        sl = channels or slice(None)
        cc = clean_full - _csub(clean_full)[None]
        dd = dirty_full - _csub(dirty_full)[None]
        velax = (hdr["CRVAL3"] + (np.arange(clean_full.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
        # `ho["folder"]` is the folder NAME, not a path: looking the .para up there finds nothing and the distance
        # silently falls back to 140 pc. The cube file's own directory is the path.
        pdir = os.path.dirname(ho["clean"])
        dist = _para_value(pdir, "distance (pc)") or float(hdr.get("DIST_PC", 140.0))
        true_incl = _para_value(pdir, "RT: imin")
        yield Case(os.path.basename(ho["folder"]), "line_emission", cc[sl], dd[sl], velax[sl],
                   abs(hdr["CDELT1"]) * 3600.0 * dist, beam_features_of(hdr),
                   true_incl if fix_incl else None, {"distance_pc": dist, "true_incl": true_incl})
        hc.close(); hd.close()


def sg_v2_case(sg_dir: str, ch0: int = 240, ch1: int = 361, channels: Optional[slice] = None) -> Case:
    """The SG v2 cube (`kinematic_data_v2`), channels 240..360 as in experiments/wiggle_domain_split.py."""
    from src.data.fits_cube_dataset import beam_features_of
    hc, hdr, cdata = _hdr_and_data(os.path.join(sg_dir, "clean_sg.fits"))
    hd, _, ddata = _hdr_and_data(os.path.join(sg_dir, "dirty_sg.fits"))
    n = cdata.shape[0]
    cc = _edge_mean(cdata)
    dc = _edge_mean(ddata)
    ch = list(range(ch0, ch1))
    velax = (hdr["CRVAL3"] + (np.array(ch) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    clean = np.stack([np.asarray(cdata[c], np.float32) for c in ch]) - cc[None]
    dirty = np.stack([np.asarray(ddata[c], np.float32) for c in ch]) - dc[None]
    sl = channels or slice(None)
    beam = beam_features_of(hdr)
    hc.close(); hd.close()
    return Case("sg_v2", "sg", clean[sl], dirty[sl], velax[sl], 4.0 / 3.0, beam, None, {"channels": (ch0, ch1)})


def sg_holdout_case(folder: str, true_incl: float = 20.0, dist_pc: float = 175.178,
                    channels: Optional[slice] = None) -> Case:
    """The SG holdout disk run_9074_00025_rt_00, whose .para states inclination 20 deg."""
    from src.data.fits_cube_dataset import beam_features_of
    name = os.path.basename(folder.rstrip("/"))
    hc, hdr, cdata = _hdr_and_data(os.path.join(folder, f"{name}_clean.fits"))
    hd, _, ddata = _hdr_and_data(os.path.join(folder, f"{name}_dirty.fits"))
    clean_full = np.asarray(cdata, np.float32)
    dirty_full = np.asarray(ddata, np.float32)
    cc = clean_full - _csub(clean_full)[None]
    dd = dirty_full - _csub(dirty_full)[None]
    velax = (hdr["CRVAL3"] + (np.arange(clean_full.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    sl = channels or slice(None)
    beam = beam_features_of(hdr)
    au = abs(hdr["CDELT1"]) * 3600.0 * dist_pc
    hc.close(); hd.close()
    return Case(name, "sg", cc[sl], dd[sl], velax[sl], au, beam, true_incl, {"distance_pc": dist_pc})


# ------------------------------------------------------------------------------------------------ #
# References computed once per case                                                                 #
# ------------------------------------------------------------------------------------------------ #
def _grad_energy(m1, sel):
    gy, gx = np.gradient(np.nan_to_num(m1))
    return float(np.mean(gx[sel] ** 2 + gy[sel] ** 2))


def _lap_var(m1):
    z = np.nan_to_num(m1)
    inner = z[1:-1, 1:-1]
    lap = z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2] - 4 * inner
    return float(lap.var())


def _corr(a, b, mask):
    ok = mask & np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else float("nan")


def _pixel_metrics(clean, dirty, pred):
    """Per-channel PSNR/SSIM on the shared DIRTY-scale normalisation (as `sweep.val_metrics`), pred clamped."""
    C = clean.shape[0]
    lo = dirty.reshape(C, -1).min(axis=1)
    hi = dirty.reshape(C, -1).max(axis=1)
    span = np.where(hi > lo, hi - lo, 1.0)
    cn = (clean - lo[:, None, None]) / span[:, None, None]
    pn = np.clip((pred - lo[:, None, None]) / span[:, None, None], 0.0, 1.0)
    mse = ((cn - pn) ** 2).reshape(C, -1).mean(axis=1)
    psnr = float(np.mean(10.0 * np.log10(1.0 / np.maximum(mse, 1e-10))))
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim = float(np.mean([ssim_fn(cn[c], pn[c], data_range=1.0) for c in range(0, C, SSIM_STRIDE)]))
    except Exception:
        ssim = float("nan")
    return psnr, ssim


@dataclass
class Prepared:
    case: Case
    m_clean: tuple
    m_dirty: tuple
    mask: np.ndarray
    geom: dict
    ref_resid: np.ndarray
    m1_clean: np.ndarray
    gradE_clean: float
    lapvar_clean: float
    dirty_resid_r: float
    psnr_dirty: float
    ssim_dirty: float
    gradE_dirty: float
    lapvar_dirty: float
    mstar_at_bound: bool
    amp_scale: float = 1.0           # factor clean was multiplied by to put it in the input's units (1.0 = untouched)
    geom_ok: bool = True             # fit converged, mass not pinned, centre on the disk (see prepare)
    geom_offset_px: float = 0.0      # fitted centre minus the M0-weighted centroid
    m1_dirty: np.ndarray = None      # quadratic M1, km/s
    resid_dirty: np.ndarray = None
    chan_idx: np.ndarray = None      # 3 channels: 20% / 50% / 80% of the cumulative clean flux (blue side, systemic, red side)
    px: np.ndarray = None            # 3 (y, x) pixels for spectra: disk peak, disk edge, off source


AMP_TOL = (0.8, 1.25)


def match_amplitude(case: Case) -> float:
    """
    Put CLEAN in the input's units when the two cubes are on different intensity scales, and return the factor (1.0 = untouched).

    The SG v2 cube's dirty is ~110 to 340x its clean. A model returns values in
    its INPUT's units, so any amplitude comparison against an unscaled clean (PSNR, M0, channel and M0 errors, calibration, invented
    structure, overshoot) measures the data's units, not the model. Velocity quantities (M1, the wiggle) are scale-free and unaffected.
    The factor is the least-squares scale of dirty on clean over the whole cube. It is applied only outside `AMP_TOL`, so a cube whose
    two halves already agree (the line-emission cubes: 1.01) is left exactly as it was and its validated numbers do not move.
    """
    c = case.clean.astype(np.float64)
    den = float((c * c).sum())
    if den <= 0:
        return 1.0
    k = float((c * case.dirty.astype(np.float64)).sum() / den)
    if AMP_TOL[0] <= k <= AMP_TOL[1] or not np.isfinite(k) or k <= 0:
        return 1.0
    case.clean = (case.clean * k).astype(np.float32)
    return k


def prepare(case: Case) -> Prepared:
    """Everything that does not depend on the checkpoint: computed once, shared by every checkpoint on this case."""
    from src.evaluation.moment_maps import generate_moment_maps, signal_mask
    from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual
    amp = match_amplitude(case)
    clean, dirty = case.clean.astype(np.float64), case.dirty.astype(np.float64)
    m_clean = generate_moment_maps("", data_velax=(clean, case.velax))
    m_dirty = generate_moment_maps("", data_velax=(dirty, case.velax))
    mask = signal_mask(m_clean[0], frac=FRAC)
    m1c = quadratic_moment1(clean, case.velax)[0] / 1000.0
    m0c = np.nan_to_num(m_clean[0])
    # `m0=` starts the fit from the EMISSION's shape. Without it the fit starts from the mask's shape, the degenerate end
    # of the mass-inclination valley, and on two line-emission cubes it ran the centre to the image edge (cy = 600.0).
    geom = fit_keplerian(m1c, mask, case.au_per_px, m0=m0c, fix_incl_deg=case.fixed_incl)
    wy, wx = np.mgrid[0:m0c.shape[0], 0:m0c.shape[1]]
    wgt = np.where(mask, np.abs(m0c), 0.0)
    cen = (float((wgt * wx).sum() / wgt.sum()), float((wgt * wy).sum() / wgt.sum())) if wgt.sum() > 0 else (geom["cx"], geom["cy"])
    offset = float(np.hypot(geom["cx"] - cen[0], geom["cy"] - cen[1]))
    extent = float(np.hypot(*(np.argwhere(mask) - np.array([cen[1], cen[0]])).T).max()) if mask.any() else 1.0
    # A pinned mass, a non-converged fit, or a centre far from the emission is a failed fit, not a measurement.
    geom_ok = bool(geom.get("success", True)) and not geom.get("mstar_at_bound", False) and offset < 0.1 * extent
    ref_resid = wiggle_residual(m1c, geom)
    m1d = quadratic_moment1(dirty, case.velax)[0] / 1000.0
    p, s = _pixel_metrics(case.clean, case.dirty, case.dirty)      # dirty scored as if it were the prediction
    resid_d = wiggle_residual(m1d, geom)
    flux = np.abs(case.clean).sum(axis=(1, 2)).astype(np.float64)
    cum = np.cumsum(flux) / max(flux.sum(), 1e-30)
    chan_idx = np.array([int(np.searchsorted(cum, q)) for q in (0.20, 0.50, 0.80)]).clip(0, case.clean.shape[0] - 1)
    m0 = np.nan_to_num(m_clean[0])
    H, W = m0.shape
    pk = np.unravel_index(int(np.argmax(m0)), m0.shape)
    inside = np.argwhere(mask & (m0 < 0.25 * m0.max()) & (m0 > 0.05 * m0.max()))
    edge = tuple(inside[len(inside) // 2]) if len(inside) else pk
    px = np.array([pk, edge, (H // 10, W // 10)])
    return Prepared(case, m_clean, m_dirty, mask, geom, ref_resid, m1c,
                    _grad_energy(m1c, mask), _lap_var(m1c),
                    _corr(ref_resid, resid_d, mask), p, s,
                    _grad_energy(m1d, mask), _lap_var(m1d), bool(geom.get("mstar_at_bound", False)),
                    amp, geom_ok, offset, m1d, resid_d, chan_idx, px)


def _artifacts(clean, dirty, pred):
    """`channel_artifacts` over every channel, on the shared dirty-scale normalisation, aggregated."""
    from src.evaluation.artifacts import channel_artifacts
    C = clean.shape[0]
    lo = dirty.reshape(C, -1).min(axis=1)
    hi = dirty.reshape(C, -1).max(axis=1)
    span = np.where(hi > lo, hi - lo, 1.0)
    fr, bl, ov, fl = [], [], [], []
    peak = float(np.max(clean.reshape(C, -1).max(axis=1) - clean.reshape(C, -1).min(axis=1)))
    for c in range(C):
        n = lambda x: (x - lo[c]) / span[c]
        try:
            a = channel_artifacts(n(clean[c]), n(dirty[c]), n(pred[c]))
        except ValueError:
            continue
        fr.append(a["invented_frac"]); bl.append(a["invented_blobs"])
        # overshoot and floor leak only mean something where the channel carries line signal
        if (clean[c].max() - clean[c].min()) > 0.05 * peak:
            ov.append(a["overshoot"]); fl.append(a["floor_leak"])
    m = lambda v: float(np.mean(v)) if v else float("nan")
    return m(fr), m(bl), m(ov), m(fl)


def invented_map(clean, dirty, pred, floor_frac=0.10, invent_frac=0.20) -> np.ndarray:
    """
    Per pixel, the fraction of channels in which the prediction asserts signal where clean has none: the same
    definition as `channel_artifacts.invented_frac` (background = clean within `floor_frac` of its span above its
    1st percentile; invented = prediction above `invent_frac` of that span), kept as a MAP so it can be looked at.
    """
    C, H, W = clean.shape
    lo = dirty.reshape(C, -1).min(axis=1)
    hi = dirty.reshape(C, -1).max(axis=1)
    span_d = np.where(hi > lo, hi - lo, 1.0)
    acc = np.zeros((H, W), np.float32)
    n = 0
    for c in range(C):
        cn = (clean[c] - lo[c]) / span_d[c]
        pn = (pred[c] - lo[c]) / span_d[c]
        floor = float(np.percentile(cn, 1))
        span = float(cn.max()) - floor
        if not np.isfinite(span) or span <= 0:
            continue
        acc += ((cn < floor + floor_frac * span) & (pn > floor + invent_frac * span)).astype(np.float32)
        n += 1
    return acc / max(n, 1)


def artifact_dict(prep: Prepared, pred: np.ndarray, m_den, m1q: np.ndarray, resid: np.ndarray) -> dict:
    """Everything the visual comparison needs from one denoised cube, small enough to keep for every checkpoint."""
    case = prep.case
    return dict(
        m0=np.asarray(m_den[0], np.float32), m1=np.asarray(m_den[1], np.float32), m2=np.asarray(m_den[2], np.float32),
        m1q=np.asarray(m1q, np.float32), resid=np.asarray(resid, np.float32),
        chan=pred[prep.chan_idx].astype(np.float32),
        spec=np.stack([pred[:, y, x] for y, x in prep.px]).astype(np.float32),
        ispec=pred.sum(axis=(1, 2)).astype(np.float32),        # whole-image flux per channel: does the model conserve it?
        invented=invented_map(case.clean, case.dirty, pred).astype(np.float16))


def reference_dict(prep: Prepared) -> dict:
    """Clean and dirty in the same form, plus the geometry, mask and axes every sheet shares."""
    case = prep.case
    d = {}
    for tag, maps, m1q, resid, cube in (("clean", prep.m_clean, prep.m1_clean, prep.ref_resid, case.clean),
                                        ("dirty", prep.m_dirty, prep.m1_dirty, prep.resid_dirty, case.dirty)):
        d.update({f"{tag}_m0": np.asarray(maps[0], np.float32), f"{tag}_m1": np.asarray(maps[1], np.float32),
                  f"{tag}_m2": np.asarray(maps[2], np.float32), f"{tag}_m1q": np.asarray(m1q, np.float32),
                  f"{tag}_resid": np.asarray(resid, np.float32), f"{tag}_chan": cube[prep.chan_idx].astype(np.float32),
                  f"{tag}_spec": np.stack([cube[:, y, x] for y, x in prep.px]).astype(np.float32),
                  f"{tag}_ispec": cube.sum(axis=(1, 2)).astype(np.float32)})
    d["dirty_invented"] = invented_map(case.clean, case.dirty, case.dirty).astype(np.float16)
    m0c = np.nan_to_num(prep.m_clean[0])
    yy, xx = np.mgrid[0:m0c.shape[0], 0:m0c.shape[1]]
    w = np.where(prep.mask, np.abs(m0c), 0.0)
    d.update(mask=prep.mask, velax=case.velax, chan_idx=prep.chan_idx, px=prep.px,
             cx=float((w * xx).sum() / max(w.sum(), 1e-30)), cy=float((w * yy).sum() / max(w.sum(), 1e-30)),   # M0 centroid, not the fit
             geom_ok=np.array(prep.geom_ok), amp_scale=np.array(prep.amp_scale), case=np.array(case.name))
    return d


def score(prep: Prepared, pred: np.ndarray) -> dict:
    """Every numeric check for one denoised cube against one prepared case."""
    from src.evaluation.moment_maps import generate_moment_maps, moment_improvement
    from src.evaluation.gi_wiggle import quadratic_moment1, wiggle_residual
    case = prep.case
    pred64 = pred.astype(np.float64)
    psnr, ssim = _pixel_metrics(case.clean, case.dirty, pred)
    m_den = generate_moment_maps("", data_velax=(pred64, case.velax))
    imp = moment_improvement(prep.m_clean, prep.m_dirty, m_den, frac=FRAC)
    m1 = quadratic_moment1(pred64, case.velax)[0] / 1000.0
    resid = wiggle_residual(m1, prep.geom)
    r = _corr(prep.ref_resid, resid, prep.mask)
    rms = lambda a: float(np.sqrt(np.nanmean(a[prep.mask] ** 2)))
    # The wiggle correlation is fooled by a residual that clean and dirty SHARE (a flat Keplerian model leaves a large one on a
    # flared, inclined disk: dirty scores 0.997 on run_0002_00560_rt_00 whatever a denoiser does). The ERROR is not: subtract
    # clean's residual and the shared part cancels. < 1 means the model brings the residual closer to clean's than dirty is.
    err_d = rms(prep.resid_dirty - prep.ref_resid)
    err_ratio = rms(resid - prep.ref_resid) / err_d if err_d > 0 else float("nan")
    inv_frac, inv_blobs, over, leak = _artifacts(case.clean, case.dirty, pred)
    extras = dict(m1q=m1, resid=resid)
    wig = lambda v: v if prep.geom_ok else float("nan")       # a failed fit is not a measurement: keep it out of every mean
    return dict(
        psnr=psnr, psnr_dirty=prep.psnr_dirty, ssim=ssim, ssim_dirty=prep.ssim_dirty,
        M0=imp["M0"], M1=imp["M1"], M2=imp["M2"], M0_all=imp["M0_all"], M1_all=imp["M1_all"],
        M2_all=imp["M2_all"], n_px=imp["n_px"],
        resid_r=wig(r), dirty_resid_r=wig(prep.dirty_resid_r), wiggle_gain=wig(r - prep.dirty_resid_r), resid_err_ratio=wig(err_ratio),
        resid_rms_ratio=wig(rms(resid) / rms(prep.ref_resid)) if rms(prep.ref_resid) > 0 else float("nan"),
        amp_scale=round(prep.amp_scale, 3), mstar_at_bound=prep.mstar_at_bound, geom_ok=prep.geom_ok, geom_offset_px=round(prep.geom_offset_px, 1),
        geom_mstar=round(float(prep.geom["mstar_msun"]), 3),
        # clean's own residual against the fitted Keplerian, km/s. Large means the flat-disk model, not a wiggle, dominates
        # the residual, so clean and dirty correlate whatever the denoiser does (run_0002_00560_rt_00: dirty resid_r 0.997)
        ref_resid_rms=round(rms(prep.ref_resid), 3),
        gradE_ratio=_grad_energy(m1, prep.mask) / prep.gradE_clean,
        gradE_ratio_dirty=prep.gradE_dirty / prep.gradE_clean,
        lapvar_ratio=_lap_var(m1) / prep.lapvar_clean,
        lapvar_ratio_dirty=prep.lapvar_dirty / prep.lapvar_clean,
        invented_frac=inv_frac, invented_blobs=inv_blobs, overshoot=over, floor_leak=leak,
    ), m_den, extras


def save_moment_figure(prep: Prepared, m_den, path: str, tag: str) -> None:
    """The clean / dirty / denoised moment-map panel (`plot_moment_comparison`), one honest scale."""
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluation.moment_maps import plot_moment_comparison
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plot_moment_comparison(prep.m_clean, prep.m_dirty, m_den, save_path=path, tag=tag)
    import matplotlib.pyplot as plt
    plt.close("all")


# ------------------------------------------------------------------------------------------------ #
# Runner                                                                                            #
# ------------------------------------------------------------------------------------------------ #
def compatible(spec: Spec, case: Case, diffusion_cases: Sequence[str]) -> Optional[str]:
    """None if this checkpoint can be scored on this case, otherwise the reason it is skipped."""
    if not spec.supported:
        return spec.why_not
    if spec.kind == "diffusion" and case.name not in diffusion_cases:
        return "diffusion sampling is ~100x a U-Net forward; limited to `diffusion_cases`"
    return None


def evaluate_case(case: Case, specs: Sequence[Spec], device: str, done: set, on_row, *,
                  figure_dir: Optional[str] = None, panel_cases: Sequence[str] = (),
                  figure_cases: Optional[Sequence[str]] = None, map_dir: Optional[str] = None,
                  diffusion_cases: Sequence[str] = (), sampling_steps: int = 25, n_avg: int = 1,
                  deadline: Optional[float] = None, batch: int = 8, log=print) -> int:
    """
    Score every compatible checkpoint on one case. `done` holds (label, case) pairs already scored;
    `on_row(row)` is called the moment a row exists, so a killed run loses nothing.
    Stops cleanly at `deadline` (epoch seconds) instead of dying mid-arm. Returns rows written.
    """
    todo = [s for s in specs if (s.label, case.name) not in done]
    if not todo:
        return 0
    log(f"\n{'=' * 78}\n{case.name}  [{case.domain}, {case.clean.shape[0]} ch]  {len(todo)} checkpoint(s) to score\n{'=' * 78}")
    t0 = time.time()
    prep = prepare(case)
    flag = ("" if prep.geom_ok else f"  !! GEOMETRY FIT FAILED (centre {prep.geom_offset_px:.0f} px off the disk, mstar "
            f"{prep.geom['mstar_msun']:.2f}{', pinned at bound' if prep.mstar_at_bound else ''}): wiggle numbers on this case are blanked")
    if prep.amp_scale != 1.0:
        log(f"  clean rescaled x{prep.amp_scale:.3g} to the input's units (dirty and clean were on different intensity scales)")
    log(f"  references ready in {time.time() - t0:.0f}s | dirty: resid_r {prep.dirty_resid_r:.4f}, "
        f"PSNR {prep.psnr_dirty:.2f}, gradE/clean {prep.gradE_dirty / prep.gradE_clean:.3f}{flag}")
    n = 0
    # maps for the visual sheets: every case unless `figure_cases` names a subset (None = all)
    keep_maps = bool(map_dir) and (figure_cases is None or case.name in figure_cases)
    if keep_maps:
        os.makedirs(map_dir, exist_ok=True)
        ref_path = os.path.join(map_dir, f"{case.name}__REF.npz")
        if not os.path.exists(ref_path):
            np.savez_compressed(ref_path, **reference_dict(prep))
    for spec in todo:
        if deadline and time.time() > deadline:
            log(f"  time budget reached, {len(todo) - n} checkpoint(s) deferred on {case.name}")
            break
        why = compatible(spec, case, diffusion_cases)
        if why:
            log(f"  --- {spec.label}: skipped ({why})")
            continue
        t1 = time.time()
        try:
            den = Denoiser(spec, device, sampling_steps=sampling_steps, n_avg=n_avg)
            pred = den(case.dirty, case.beam_vec, batch=batch)
            res, m_den, extras = score(prep, pred)
        except Exception as e:                       # one bad checkpoint must not end the session
            log(f"  --- {spec.label}: FAILED {type(e).__name__}: {str(e)[:160]}")
            on_row({"checkpoint": spec.label, "source": spec.source, "family": spec.family, "case": case.name,
                    "domain": case.domain, "note": f"FAILED {type(e).__name__}: {str(e)[:120]}"})
            continue
        row = {"checkpoint": spec.label, "source": spec.source, "family": spec.family, "case": case.name,
               "domain": case.domain, "n_channels": case.clean.shape[0], **res,
               "wall_s": round(time.time() - t1, 1), "note": "" if prep.geom_ok else "geom_failed: wiggle blanked"}
        on_row(row)
        n += 1
        log(f"  {spec.label:38s} PSNR {res['psnr']:6.2f} ({res['psnr'] - res['psnr_dirty']:+.2f})  "
            f"M0 {res['M0']:+7.1f} M1 {res['M1']:+7.1f} M2 {res['M2']:+7.1f}  resid_r {res['resid_r']:.3f} "
            f"gradE {res['gradE_ratio']:.2f}  blobs {res['invented_blobs']:.1f}  ({time.time() - t1:.0f}s)")
        if keep_maps:
            np.savez_compressed(os.path.join(map_dir, f"{case.name}__{spec.label}.npz"),
                                **artifact_dict(prep, pred, m_den, extras["m1q"], extras["resid"]))
        if figure_dir and case.name in panel_cases:
            save_moment_figure(prep, m_den, os.path.join(figure_dir, f"{case.name}__{spec.label}.png"),
                               f"{spec.label} on {case.name}")
        del den, pred
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return n
