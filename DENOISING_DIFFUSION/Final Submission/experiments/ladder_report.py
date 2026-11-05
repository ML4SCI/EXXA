"""
SNR-ladder table from eval JSONs written by train_unified.py --eval-ladder.
  python experiments/ladder_report.py name=eval_x.json [name2=...]
Per peak SNR: GI disc PSNR / M0 / M1 / M2 / rc_r (dirty and Wiener of that file for reference) and the Keplerian twin's spurious rotation-curve rms.
Moments are signal-masked, noise-clipped improvements over dirty. One simulation, so no spread; the Wiener row is the one stored in the same file.
"""
import json, re, sys

def main():
    models = dict(a.split("=", 1) for a in sys.argv[1:])
    R = {n: json.load(open(p)) for n, p in models.items()}
    snrs = sorted({float(m.group(1)) for r in R.values() for k in r for m in [re.match(r"speedie_ladder_GI_r0\.5_snr([0-9.]+)@", k)] if m})
    print("| peak SNR | model | PSNR (dirty) | M0 % | M1 % | M2 % | rc_r (dirty / wiener) | twin spurious m/s |\n|---|---|---|---|---|---|---|---|")
    for snr in snrs:
        for n, r in R.items():
            gi = next((v for k, v in r.items() if k.startswith(f"speedie_ladder_GI_r0.5_snr{snr:g}@") and "failed" not in v), None)
            kp = next((v for k, v in r.items() if k.startswith(f"speedie_ladder_Kep_r0.5_snr{snr:g}@") and "failed" not in v), None)
            if gi is None:
                print(f"| {snr:g} | {n} | (missing) | | | | | |"); continue
            m, g = gi["model"], gi["wiggle"]
            sp = kp["wiggle"]["model"].get("rc_rms_ms", float("nan")) - kp["wiggle"]["model"].get("rc_rms_clean_ms", 0.0) if kp else float("nan")
            print(f"| {snr:g} | {n} | {m['psnr']:.1f} ({gi['dirty_psnr']:.1f}) | {m['M0']:.0f} | {m['M1']:.0f} | {m['M2']:.0f} | "
                  f"{g['model']['rc_r']:.2f} ({g['dirty']['rc_r']:.2f} / {g['wiener']['rc_r']:.2f}) | {sp:+.1f} |")
        w = gi["wiener"] if gi else None
        if w:
            print(f"| {snr:g} | wiener (file) | {w['psnr']:.1f} | {w['M0']:.0f} | {w['M1']:.0f} | {w['M2']:.0f} | | |")

if __name__ == "__main__":
    main()
