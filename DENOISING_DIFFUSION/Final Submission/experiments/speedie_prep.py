"""
Prepare the Speedie et al. 2024 simulated GI disk (Zenodo 11668694, CC-BY-4.0) as truth + noisy pairs for evaluation and training.

What the release holds (README, headers read 2026-10-10):
  GIdisk_13CO_lines.contsub.fits / Kepdisk_13CO_lines.contsub.fits   MCFOST model cubes, 2048 x 2048 x 201 at 0.02"/px, 42 m/s,
        JY/PIXEL, NO beam. The GI disk (Phantom SPH, spiral waves) and its Keplerian twin from the same dump (no radial / vertical velocity).
  {GI,Kep}disk_13CO_robust{0.5,1.5}.zip :: lines.contsub.JvMcorr.pbcor.fits   the model run through syndisk.synthetic_ALMA to the
        beam, channel width and rms of ALMA program 2021.1.00690.S (AB Aur): 2048 x 2048 x 200 at 42 m/s, JY/BEAM, first channel at -4179 m/s,
        i.e. half a channel off the model grid, robust 0.5 beam 0.237" x 0.175".
So  dirty = the synthetic ALMA cube (real-like noise, whatever spectral covariance syndisk gave it)
    clean = model, crop, beam-convolved with the synthetic cube's own beam, Jy/pixel -> Jy/beam, resampled to its velocity grid.
This is the noise-only regime of a real ALMA CLEAN image (the beam is inside clean), and the Keplerian twin is the false-wiggle test.
Geometry from the release's own MCFOST parameter file: distance 155.9 pc, inclination 23.2 deg, PA 236.7 deg (corrected 2026-10-10;
the first version wrote 162.9 pc / 23.5 deg). Not exoALMA data (AB Aur program 2021.1.00690.S, PI Dong; simulated here), so allowed by the project's rule of never training on exoALMA data.

The disk is larger than 600 px at 0.02"/px (the systemic channel spans the whole 600 px vertically), so the default crop is 1024.
The zips are deflate-compressed, so the synthetic cubes are STREAMED (3.2 GB each); the model cubes are read by range requests, one
contiguous row block per channel. Run on Kaggle (internet on, CPU only): locally Zenodo gave 0.2 MB/s.

Run: python experiments/speedie_prep.py --out /kaggle/working/speedie [--crop 1024] [--which GI,Kep] [--robust 0.5,1.5]
"""
import io, os, sys, json, time, argparse, zipfile, urllib.request
import concurrent.futures as cf
import numpy as np

BASE = "https://zenodo.org/records/11668694/files/"


# ---------------------------------------------------------------------------------------------------- http
class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a big read-ahead block so zipfile does not issue 100k tiny requests."""

    def __init__(self, url, block=32 << 20):
        self.url, self.block, self.pos = url, block, 0
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120)
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self._buf, self._start = b"", 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _fetch(self, start, n):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + n - 1}"})
                return urllib.request.urlopen(req, timeout=300).read()
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise IOError(f"range fetch failed: {self.url} bytes {start}-{start + n - 1}")

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        while n > 0:
            if not (self._start <= self.pos < self._start + len(self._buf)):
                self._start = self.pos
                self._buf = self._fetch(self.pos, min(max(self.block, n if n < (256 << 20) else self.block), self.size - self.pos))
            off = self.pos - self._start
            take = min(n, len(self._buf) - off)
            out += self._buf[off:off + take]
            self.pos += take
            n -= take
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def range_get(url, start, n, tries=8):
    """One range request with retries and backoff (Zenodo throttles PER CONNECTION: 8 parallel requests ran 6.6x faster than 1, 2026-10-10)."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + n - 1}"})
            d = urllib.request.urlopen(req, timeout=300).read()
            if len(d) == n:
                return d
        except Exception:
            pass
        time.sleep(min(60, 2 ** attempt))
    raise IOError(f"range fetch failed after {tries} tries: {url} bytes {start}-{start + n - 1}")


def remote_size(url):
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120)
    return int(r.headers["Content-Range"].split("/")[1])


def download_parallel(url, dst, workers=16, chunk=8 << 20):
    """Download `url` to `dst` with `workers` concurrent range requests, each written at its own offset. Resumes: chunks already on disk
    (tracked in dst + '.done') are skipped, so a killed session does not restart a 3 GB file."""
    size = remote_size(url)
    done_path = dst + ".done"
    done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
    if not os.path.exists(dst) or os.path.getsize(dst) != size:
        with open(dst, "wb") as f:
            f.truncate(size)
        done = set()
    starts = [o for o in range(0, size, chunk) if o not in done]
    t0, got = time.time(), [0]
    lock = __import__("threading").Lock()

    def work(o):
        n = min(chunk, size - o)
        d = range_get(url, o, n)
        with lock:
            with open(dst, "r+b") as f:
                f.seek(o)
                f.write(d)
            with open(done_path, "a") as f:
                f.write(f"{o}\n")
            got[0] += n
            if (got[0] // chunk) % 40 == 0:
                print(f"    {os.path.basename(dst)}: {100 * (size - len(starts) * chunk + got[0]) / size:.0f}% ({got[0] / 1e6 / (time.time() - t0):.2f} MB/s)", flush=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, starts))
    assert os.path.getsize(dst) == size
    os.remove(done_path) if os.path.exists(done_path) else None
    return size


def parse_fits_header(raw: bytes):
    """(dict of cards, header length in bytes) from the first FITS blocks."""
    cards, end = {}, None
    for i in range(0, len(raw), 80):
        c = raw[i:i + 80].decode("ascii", "replace")
        if c.startswith("END"):
            end = i + 80
            break
        if "=" in c[:10]:
            v = c[10:].split("/")[0].strip().strip("'").strip()
            cards[c[:8].strip()] = v
    return cards, ((end + 2879) // 2880) * 2880


def _crop_slices(n, crop):
    c0 = (n - crop) // 2
    return slice(c0, c0 + crop)


# ---------------------------------------------------------------------------------------------------- readers
def stream_zip_cube(zip_url, member_suffix, crop):
    """Stream a deflate-compressed FITS cube inside a remote zip; return (header cards, (C, crop, crop) float32 array, centred crop)."""
    f = RangeFile(zip_url) if str(zip_url).startswith("http") else open(zip_url, "rb")
    zf = zipfile.ZipFile(f)
    info = [i for i in zf.infolist() if i.filename.endswith(member_suffix)][0]
    with zf.open(info) as m:
        head = m.read(2880 * 6)
        h, hsz = parse_fits_header(head)
        rest = m.read(hsz - len(head)) if hsz > len(head) else b""
        carry = head[hsz:] if hsz <= len(head) else b""
        C, H, W = int(h["NAXIS3"]), int(h["NAXIS2"]), int(h["NAXIS1"])
        assert h["BITPIX"] == "-32", h["BITPIX"]
        plane_bytes = H * W * 4
        sl_y, sl_x = _crop_slices(H, crop), _crop_slices(W, crop)
        out = np.empty((C, crop, crop), np.float32)
        buf = carry
        for c in range(C):
            while len(buf) < plane_bytes:
                chunk = m.read(64 << 20)
                if not chunk:
                    raise IOError(f"stream ended at channel {c} of {C}")
                buf += chunk
            plane = np.frombuffer(buf[:plane_bytes], dtype=">f4").reshape(H, W)
            out[c] = plane[sl_y, sl_x]
            buf = buf[plane_bytes:]
            if c % 25 == 0:
                print(f"    {member_suffix[:28]} channel {c}/{C}", flush=True)
    return h, out


def read_model_crop(url, crop, workers=16):
    """Model cube (uncompressed FITS on Zenodo): per channel, the contiguous block of `crop` rows, then the column crop; channels fetched in parallel."""
    head = range_get(url, 0, 2880 * 4)
    h, hsz = parse_fits_header(head)
    C, H, W = int(h["NAXIS3"]), int(h["NAXIS2"]), int(h["NAXIS1"])
    sl_y, sl_x = _crop_slices(H, crop), _crop_slices(W, crop)
    out = np.empty((C, crop, crop), np.float32)
    n_done = [0]

    def one(c):
        off = hsz + (c * H + sl_y.start) * W * 4
        rows = np.frombuffer(range_get(url, off, crop * W * 4), dtype=">f4").reshape(crop, W)
        out[c] = rows[:, sl_x]
        n_done[0] += 1
        if n_done[0] % 25 == 0:
            print(f"    model channel {n_done[0]}/{C}", flush=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(C)))
    return h, out


# ---------------------------------------------------------------------------------------------------- physics
def beam_kernel(bmaj_deg, bmin_deg, bpa_deg, pix_deg, cdelt1_sign=-1, half=None):
    """Unit-sum elliptical Gaussian (FWHM bmaj x bmin, PA east of north) on the pixel grid [row = y (north), col = x (east is -x if CDELT1 < 0)]."""
    smaj, smin = bmaj_deg / pix_deg / 2.3548200450309493, bmin_deg / pix_deg / 2.3548200450309493
    half = half or int(np.ceil(4 * smaj))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    t = np.radians(bpa_deg)
    # unit vector of the major axis in (x, y): north is +y, east is +x only if cdelt1 > 0
    ex = np.sin(t) * (1.0 if cdelt1_sign > 0 else -1.0)
    ey = np.cos(t)
    u = x * ex + y * ey                         # along the major axis
    v = -x * ey + y * ex                        # along the minor axis
    k = np.exp(-0.5 * ((u / smaj) ** 2 + (v / smin) ** 2))
    return k / k.sum()


def jy_pixel_to_jy_beam_factor(bmaj_deg, bmin_deg, pix_deg):
    """Beam area in pixels: Jy/beam = (unit-sum-beam-smoothed Jy/pixel) * (pi / (4 ln 2)) * BMAJ * BMIN / pix^2."""
    return np.pi / (4.0 * np.log(2.0)) * bmaj_deg * bmin_deg / pix_deg ** 2


def convolve_cube(cube, kernel):
    """FFT convolution of every channel with `kernel` (centred), same shape out."""
    import scipy.fft as sfft
    n = cube.shape[-1]
    pad = np.zeros((n, n))
    h = kernel.shape[0] // 2
    c = n // 2
    pad[c - h:c + h + 1, c - h:c + h + 1] = kernel
    K = sfft.fft2(np.fft.ifftshift(pad))
    out = np.empty_like(cube, dtype=np.float32)
    for i in range(0, cube.shape[0], 16):
        out[i:i + 16] = sfft.ifft2(sfft.fft2(cube[i:i + 16].astype(np.float64), workers=-1) * K[None], workers=-1).real
    return out


def make_clean(model_jypix, model_v, syn_v, bmaj, bmin, bpa, pix_deg, cdelt1_sign=-1):
    """Model (Jy/pixel, no beam) -> the noise-free version of the synthetic cube: beam-convolved, Jy/beam, on the synthetic velocity grid."""
    conv = convolve_cube(model_jypix, beam_kernel(bmaj, bmin, bpa, pix_deg, cdelt1_sign)) * jy_pixel_to_jy_beam_factor(bmaj, bmin, pix_deg)
    out = np.empty((len(syn_v),) + conv.shape[1:], np.float32)
    for j, v in enumerate(syn_v):                 # linear interpolation in velocity (the synthetic grid sits half a channel off the model's)
        f = (v - model_v[0]) / (model_v[1] - model_v[0])
        i0 = int(np.clip(np.floor(f), 0, len(model_v) - 2))
        w = float(np.clip(f - i0, 0.0, 1.0))
        out[j] = (1 - w) * conv[i0] + w * conv[i0 + 1]
    return out


def noise_report(dirty, clean, edge=12):
    """Measured properties of the synthetic cube's noise: the part that matters for our noise model."""
    C = dirty.shape[0]
    res = dirty - clean
    line_free = np.r_[0:edge, C - edge:C]
    r = res[line_free].astype(np.float64)
    sig = float(r.std())
    # spectral covariance of the noise: correlation between channels c and c+lag, in line-free channels
    lags = {}
    for lag in (1, 2, 3, 5):
        a, b = r[:-lag].ravel(), r[lag:].ravel()
        lags[lag] = float(np.corrcoef(a, b)[0, 1]) if len(line_free) > lag else float("nan")
    # spatial correlation: lag-1 and lag-3 along x within a channel
    sp = {l: float(np.corrcoef(r[:, :, :-l].ravel(), r[:, :, l:].ravel())[0, 1]) for l in (1, 3, 8)}
    # does the clean prediction explain the signal channels? residual rms in the brightest channels vs line-free noise
    peak_ch = np.argsort(clean.reshape(C, -1).max(1))[-10:]
    sig_res = float(res[peak_ch].std())
    rel_err = float(np.sqrt(np.mean(res[peak_ch] ** 2)) / np.sqrt(np.mean(dirty[peak_ch] ** 2)))
    return {"noise_rms_jybeam": sig, "spectral_corr_lag": lags, "spatial_corr_x_lag": sp,
            "residual_std_brightest_channels": sig_res, "residual_over_line_free_noise": sig_res / sig,
            "peak_clean_jybeam": float(clean.max()), "peak_snr": float(clean.max() / sig),
            "brightest_channel_rel_residual": rel_err}


def write_cube(path, data, hdr_cards, extra=None):
    from astropy.io import fits
    h = fits.Header()
    for k in ("CDELT1", "CDELT2", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3", "RESTFREQ", "BUNIT", "BMAJ", "BMIN", "BPA"):
        if k in hdr_cards:
            try:
                h[k] = float(hdr_cards[k])
            except ValueError:
                h[k] = hdr_cards[k]
    for k, v in (extra or {}).items():
        h[k] = v
    fits.PrimaryHDU(data.astype(np.float32), h).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="", help="where the model-cube crops are kept (default: --out). Keep it OUT of a notebook's Output.")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--which", default="GI,Kep")
    ap.add_argument("--robust", default="0.5,1.5")
    ap.add_argument("--workers", type=int, default=16, help="parallel range requests (Zenodo throttles per connection)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = a.cache or a.out
    os.makedirs(cache, exist_ok=True)
    report_path = os.path.join(a.out, "speedie_report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    models = {}
    t_all = time.time()
    for rb in a.robust.split(","):                       # robust 0.5 for BOTH disks first: the twin matters more than the second weighting
        for which in a.which.split(","):
            key = f"{which}_r{rb}"
            if key in report and os.path.exists(os.path.join(a.out, f"{key}_dirty.fits")):
                print(f"[{key}] done, skipped", flush=True)
                continue
            t0 = time.time()
            if which not in models:
                mpath = os.path.join(cache, f"{which}_model_crop.npy")
                url = BASE + f"{which}disk_13CO_lines.contsub.fits?download=1"
                if os.path.exists(mpath):
                    model = np.load(mpath)
                else:
                    print(f"[{which}] model cube (parallel x{a.workers})", flush=True)
                    _, model = read_model_crop(url, a.crop, a.workers)
                    np.save(mpath, model)
                cd = float(parse_fits_header(range_get(url, 0, 11520))[0]["CDELT3"])
                models[which] = (model, (np.arange(model.shape[0]) - 100) * cd / 1000.0)
            model, model_v = models[which]
            zpath = os.path.join(cache, f"{which}disk_13CO_robust{rb}.zip")
            print(f"[{key}] downloading the synthetic ALMA zip (parallel x{a.workers})", flush=True)
            download_parallel(BASE + f"{which}disk_13CO_robust{rb}.zip?download=1", zpath, a.workers)
            print(f"[{key}] unpacking + building truth ({(time.time() - t0) / 60:.0f} min so far)", flush=True)
            h, dirty = stream_zip_cube(zpath, "lines.contsub.JvMcorr.pbcor.fits", a.crop)
            os.remove(zpath)
            syn_v = (float(h["CRVAL3"]) + (np.arange(dirty.shape[0]) + 1 - float(h["CRPIX3"])) * float(h["CDELT3"])) / 1000.0
            pix = abs(float(h["CDELT1"]))
            clean = make_clean(model, model_v, syn_v, float(h["BMAJ"]), float(h["BMIN"]), float(h["BPA"]), pix)
            rep = noise_report(dirty, clean)
            rep.update(bmaj_arcsec=float(h["BMAJ"]) * 3600, bmin_arcsec=float(h["BMIN"]) * 3600, pix_arcsec=pix * 3600,
                       n_chan=int(dirty.shape[0]), crop=a.crop, dv_kms=float(syn_v[1] - syn_v[0]), minutes=(time.time() - t0) / 60)
            report[key] = rep
            extra = {"SPD_WHICH": which, "SPD_ROBUST": float(rb), "DIST_PC": 155.9, "INCL_DEG": 23.2, "PA_DEG": 236.7}
            write_cube(os.path.join(a.out, f"{key}_dirty.fits"), dirty, h, extra)
            write_cube(os.path.join(a.out, f"{key}_clean.fits"), clean, h, extra)
            json.dump(report, open(report_path, "w"), indent=1)
            print(f"[{key}] rms {rep['noise_rms_jybeam']:.3e} Jy/beam, peak SNR {rep['peak_snr']:.1f}, spectral lag-1 corr "
                  f"{rep['spectral_corr_lag'][1]:+.2f}, clean explains signal channels to {100*rep['brightest_channel_rel_residual']:.0f}% rms "
                  f"[{(time.time() - t_all) / 60:.0f} min total]", flush=True)
            del dirty, clean
    print("SPEEDIE PREP DONE", flush=True)


if __name__ == "__main__":
    main()
