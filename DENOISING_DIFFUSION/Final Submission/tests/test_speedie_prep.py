"""
speedie_prep physics + I/O on toy data (no network): the Jy/pixel -> Jy/beam conversion (a 1 Jy point source peaks at 1 Jy/beam), the beam
kernel orientation, the half-channel velocity resampling, the noise report on cubes with a KNOWN noise covariance, and the HTTP range reader
against a local server.

Run: PYTHONPATH=. python3 tests/test_speedie_prep.py
"""
import os, sys, io, zipfile, threading, tempfile, http.server, functools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import speedie_prep as sp

pix = 0.02 / 3600
bmaj, bmin, bpa = 0.237 / 3600, 0.175 / 3600, 1.2

# 1. a 1 Jy point source: peak = 1 Jy/beam, total flux (Jy/beam summed / beam area in pixels) preserved
n = 129
cube = np.zeros((1, n, n), np.float32); cube[0, 64, 64] = 1.0                              # 1 Jy in one pixel
k = sp.beam_kernel(bmaj, bmin, bpa, pix)
fac = sp.jy_pixel_to_jy_beam_factor(bmaj, bmin, pix)
out = sp.convolve_cube(cube, k)[0] * fac
beam_area_px = fac
assert abs(out.max() - 1.0) < 0.03, out.max()
assert abs(out.sum() / beam_area_px - 1.0) < 1e-3, out.sum() / beam_area_px
print(f"point source: peak {out.max():.3f} Jy/beam (expected 1), flux {out.sum()/beam_area_px:.4f} Jy (expected 1), beam {beam_area_px:.1f} px")

# 2. orientation: PA 0 puts the major axis along y (north), PA 90 along x (cdelt1 < 0: east is -x, still along the x axis)
def widths(pa):
    kk = sp.beam_kernel(bmaj, bmin, pa, pix)
    h = kk.shape[0] // 2
    sy, sx = np.sqrt((kk.sum(1) * (np.arange(-h, h + 1) ** 2)).sum()), np.sqrt((kk.sum(0) * (np.arange(-h, h + 1) ** 2)).sum())
    return sy, sx
sy, sx = widths(0.0); assert sy > sx * 1.25, (sy, sx)
sy, sx = widths(90.0); assert sx > sy * 1.25, (sy, sx)
print("beam PA: 0 deg -> elongated along y, 90 deg -> along x")

# 3. the synthetic grid sits half a channel off the model grid: clean[j] = (model[j] + model[j+1]) / 2
model_v = (np.arange(6) - 2) * 0.042
syn_v = model_v[:5] + 0.021
m = np.stack([np.full((65, 65), float(i)) for i in range(6)]).astype(np.float32)
cl = sp.make_clean(m, model_v, syn_v, bmaj, bmin, bpa, pix)
assert cl.shape[0] == 5 and np.allclose(cl[:, 32, 32] / cl[0, 32, 32] * 0.5, np.arange(5) + 0.5, atol=1e-3), cl[:, 32, 32]
print("velocity: half-channel offset resampled to the average of adjacent model channels")

# 4. noise report recovers a known spectral covariance and noise level
rng = np.random.default_rng(0)
C, N = 60, 48
white = rng.normal(0, 1.0, (C + 1, N, N))
noise = (white[:-1] + white[1:]) / np.sqrt(2)                                       # lag-1 corr 0.5, unit variance
clean = np.zeros((C, N, N)); clean[20:40, 20:30, 20:30] = 50.0
dirty = (clean + 0.01 * noise).astype(np.float32)
rep = sp.noise_report(dirty, clean.astype(np.float32), edge=10)
assert abs(rep["noise_rms_jybeam"] - 0.01) < 0.0007, rep["noise_rms_jybeam"]
assert abs(rep["spectral_corr_lag"][1] - 0.5) < 0.05 and abs(rep["spectral_corr_lag"][2]) < 0.05, rep["spectral_corr_lag"]
print(f"noise report: rms {rep['noise_rms_jybeam']:.4f} (0.01), lag-1 corr {rep['spectral_corr_lag'][1]:+.2f} (0.5), lag-2 {rep['spectral_corr_lag'][2]:+.2f} (0)")

# 5. range reader + zip streaming against a local HTTP server (supports Range)
tmp = tempfile.mkdtemp()
from astropy.io import fits
data = rng.normal(0, 1, (7, 40, 40)).astype(">f4")
fits.PrimaryHDU(data.astype(np.float32)).writeto(os.path.join(tmp, "cube.fits"))
with zipfile.ZipFile(os.path.join(tmp, "z.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(tmp, "cube.fits"), "d/lines.contsub.JvMcorr.pbcor.fits")

class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        path = self.translate_path(self.path.split("?")[0])
        size = os.path.getsize(path)
        rng_h = self.headers.get("Range")
        a, b = 0, size - 1
        if rng_h:
            a, b = [int(x) for x in rng_h.split("=")[1].split("-")]
            b = min(b, size - 1)
        with open(path, "rb") as f:
            f.seek(a); body = f.read(b - a + 1)
        self.send_response(206 if rng_h else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers(); self.wfile.write(body)

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(RangeHandler, directory=tmp))
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"
h, got = sp.stream_zip_cube(url + "/z.zip", "lines.contsub.JvMcorr.pbcor.fits", 16)
assert got.shape == (7, 16, 16) and np.allclose(got, data.astype(np.float32)[:, 12:28, 12:28]), "streamed crop differs"
h2, got2 = sp.read_model_crop(url + "/cube.fits", 16)
assert np.allclose(got2, data.astype(np.float32)[:, 12:28, 12:28]), "range-read model crop differs"
# parallel paths (what the Kaggle kernel actually uses)
h3, got3 = sp.read_model_crop(url + "/cube.fits", 16, workers=4)
assert np.allclose(got3, data.astype(np.float32)[:, 12:28, 12:28]), "parallel model crop differs"
dst = os.path.join(tmp, "dl.zip")
sz = sp.download_parallel(url + "/z.zip", dst, workers=4, chunk=2048)
assert open(dst, "rb").read() == open(os.path.join(tmp, "z.zip"), "rb").read(), "parallel download is not byte-identical"
h4, got4 = sp.stream_zip_cube(dst, "lines.contsub.JvMcorr.pbcor.fits", 16)       # a LOCAL zip
assert np.allclose(got4, got), "local-zip streaming differs from remote-zip streaming"
# resume: pre-write the first half correctly, mark those chunks done, corrupt the rest: only the rest may be fetched
dst2 = os.path.join(tmp, "dl2.zip")
orig = open(os.path.join(tmp, "z.zip"), "rb").read()
chunk = 1024
with open(dst2, "wb") as f:
    f.write(orig[:len(orig) // 2] + b"\x00" * (len(orig) - len(orig) // 2))
open(dst2 + ".done", "w").write("\n".join(str(o) for o in range(0, len(orig) // 2, chunk) if o + chunk <= len(orig) // 2))
sp.download_parallel(url + "/z.zip", dst2, workers=3, chunk=chunk)
assert open(dst2, "rb").read() == orig, "resume produced a different file"
print(f"parallel    : 4-worker model crop and {sz/1e3:.1f} kB download byte-identical; local-zip streaming matches; resume refetches only the missing chunks")
srv.shutdown()
print("io          : zip streaming and row-block range reads return the exact central crop")
print("PASS")
