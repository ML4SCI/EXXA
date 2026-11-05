"""
Detection limit of the minor-axis wiggle: for every held-out GI case in the archived evaluations, the clean wiggle size (PV magnitude, 5 px slit,
m/s), the SNR (named peak SNR for the Speedie ladders, else the blind estimate snr_est), and how well the model, Wiener and dirty recover the PV
curve (PV r, 5 px) and the 2D wiggle (rc_r). Light: reads eval JSONs only.
Usage: python experiments/detection_limit.py <eval json> [...]
"""
import json, re, sys

rows = []
for p in sys.argv[1:]:
    r = json.load(open(p))
    for k, v in r.items():
        if not isinstance(v, dict) or "wiggle" not in v or "Kep" in k or "9074" in k or "9025" in k or "sgv2" in k or k.startswith("run_00") or "_ai" in k:
            continue
        if not (k.endswith("@600") or k.endswith("@1024")):
            continue
        w = v["wiggle"]
        m = re.search(r"snr(\d+(?:\.\d+)?)", k)
        snr = float(m.group(1)) if m else v.get("snr_est")
        rows.append((w["clean"].get("pv5_mag_ms"), snr, "peak" if m else "blind", k, w["model"].get("pv5_r"), w["wiener"].get("pv5_r"), w["dirty"].get("pv5_r"),
                     w["model"]["rc_r"], w["wiener"]["rc_r"]))
rows.sort(key=lambda x: (x[0] or 0))
print("| case | clean PV magnitude (m/s) | SNR | PV5 r model / Wiener / dirty | rc_r model / Wiener |")
print("|---|---|---|---|---|")
f = lambda x: "nan" if x is None else f"{x:.2f}"
for mag, snr, kind, k, pm, pw, pd, rm, rw in rows:
    print(f"| {k} | {mag:.0f} | {snr:.0f} ({kind}) | {f(pm)} / {f(pw)} / {f(pd)} | {f(rm)} / {f(rw)} |")
