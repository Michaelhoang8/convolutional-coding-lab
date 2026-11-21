"""Render plots and a static report from the preserved measured run."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"uncoded": "#596d7b", "hard": "#bf6027", "soft": "#1959d6"}
LABELS = {"uncoded": "Uncoded BPSK", "hard": "Hard Viterbi", "soft": "Soft Viterbi"}


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#102537",
            "text.color": "#102537",
            "xtick.color": "#526778",
            "ytick.color": "#526778",
            "axes.edgecolor": "#c3d1da",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#9eafbb",
            "grid.linewidth": 0.7,
        }
    )


def plots(output, summary, demo):
    configure()
    fig, ax = plt.subplots(figsize=(10.8, 5.9), layout="constrained")
    for method in COLORS:
        group = [row for row in summary if row["method"] == method]
        positive = [row for row in group if float(row["information_ber"]) > 0]
        ax.semilogy(
            [float(row["ebn0_db_information"]) for row in positive],
            [float(row["information_ber"]) for row in positive],
            "o-",
            color=COLORS[method],
            linewidth=2.4,
            markersize=7,
            label=LABELS[method],
        )
    uncoded = [row for row in summary if row["method"] == "uncoded"]
    ax.semilogy(
        [float(r["ebn0_db_information"]) for r in uncoded],
        [float(r["uncoded_bpsk_theory_ber"]) for r in uncoded],
        linestyle="--",
        color="#a6b3bc",
        linewidth=1.6,
        label="Uncoded AWGN theory",
    )
    soft_zero = [
        r for r in summary if r["method"] == "soft" and float(r["information_ber"]) == 0
    ]
    if soft_zero:
        row = soft_zero[-1]
        ax.text(
            0.02,
            0.04,
            f"Soft at {float(row['ebn0_db_information']):g} dB: 0 / {int(row['information_bits']):,} errors\nZero is omitted from the logarithmic axis.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color=COLORS["soft"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        )
    ax.set(
        xlabel="Eb(info) / N0 (dB), including tail-symbol energy",
        ylabel="Information-bit error rate",
        xticks=[0, 1.5, 3, 4.5, 6],
        ylim=(1e-5, 0.4),
    )
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(output / "information-ber.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.5), layout="constrained")
    for method in COLORS:
        group = [r for r in summary if r["method"] == method]
        x = np.array([float(r["ebn0_db_information"]) for r in group])
        y = np.array([float(r["block_error_rate"]) for r in group])
        lower = y - np.array([float(r["block_wilson95_low"]) for r in group])
        upper = np.array([float(r["block_wilson95_high"]) for r in group]) - y
        ax.errorbar(
            x,
            y,
            yerr=[lower, upper],
            fmt="o-",
            color=COLORS[method],
            linewidth=2.2,
            capsize=4,
            label=LABELS[method],
        )
    ax.set(
        xlabel="Eb(info) / N0 (dB)",
        ylabel="Frame-error probability estimate",
        xticks=[0, 1.5, 3, 4.5, 6],
        ylim=(-0.025, 1.035),
    )
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output / "block-errors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 3.7), layout="constrained")
    samples = np.array(demo["received"])
    truth = np.array([1 - 2 * b for b in demo["encoded"]["coded"]])
    wrong = np.array(demo["hard_sliced"]) != demo["encoded"]["coded"]
    x = np.arange(1, len(samples) + 1)
    ax.vlines(
        x,
        0,
        samples,
        colors=[COLORS["hard"] if v else COLORS["soft"] for v in wrong],
        alpha=0.55,
        linewidth=2,
    )
    ax.scatter(
        x[~wrong],
        samples[~wrong],
        c=COLORS["soft"],
        label="Correct sliced sign",
        zorder=3,
    )
    ax.scatter(
        x[wrong],
        samples[wrong],
        c=COLORS["hard"],
        label="Incorrect sliced sign",
        zorder=3,
    )
    ax.scatter(
        x,
        truth,
        facecolors="none",
        edgecolors="#596d7b",
        marker="s",
        label="Transmitted BPSK level",
        zorder=3,
    )
    ax.axhline(0, color="#596d7b", linewidth=1)
    ax.set(
        xlabel="Coded symbol index in the illustrative frame",
        ylabel="Received amplitude (Es = 1)",
        xticks=range(1, 21),
    )
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=9)
    fig.savefig(output / "received-samples.png", dpi=160)
    plt.close(fig)


def write_report(path, summary, environment, demo):
    error_rows = []
    for snr in environment["snr_points_db"]:
        group = [r for r in summary if float(r["ebn0_db_information"]) == snr]
        cells = "".join(
            f"<td>{int(r['information_bit_errors']):,}<small>{float(r['information_ber']):.6g}</small></td>"
            for r in group
        )
        error_rows.append(f"<tr><th>{snr:g} dB</th>{cells}</tr>")
    block_rows = []
    for method in COLORS:
        row = next(
            r
            for r in summary
            if r["method"] == method and float(r["ebn0_db_information"]) == 6
        )
        block_rows.append(
            f"<tr><th>{LABELS[method]}</th><td>{row['block_errors']} / {row['frames']}</td><td>{100*float(row['block_wilson95_low']):.2f}–{100*float(row['block_wilson95_high']):.2f}%</td></tr>"
        )
    timing_rows = []
    for method in COLORS:
        row = next(
            r
            for r in summary
            if r["method"] == method and float(r["ebn0_db_information"]) == 3
        )
        timing_rows.append(
            f"<tr><th>{LABELS[method]}</th><td>{float(row['median_decode_ns'])/1000:.2f} µs</td><td>{int(row['min_decode_ns'])/1000:.1f}–{int(row['max_decode_ns'])/1000:.1f} µs</td></tr>"
        )
    content = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Convolutional coding lab</title>
<style>
:root{--ink:#102537;--muted:#526778;--line:#dce4e9;--blue:#1959d6;--orange:#bf6027}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#fff;color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}a{color:var(--blue);text-underline-offset:4px}nav{position:sticky;top:0;z-index:4;background:#fffffff2;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:16px max(28px,calc((100vw - 1180px)/2));display:flex;gap:25px;font-size:13px;white-space:nowrap;overflow:auto}nav a{color:var(--muted);text-decoration:none}nav a:hover{color:var(--blue)}main{max-width:1180px;margin:auto;padding:0 28px}section{padding:82px 0;border-bottom:1px solid var(--line);scroll-margin-top:54px}h1{font-size:clamp(44px,6.4vw,82px);font-weight:540;line-height:1.06;letter-spacing:-.055em;margin:18px 0 25px}h2{font-size:35px;font-weight:530;letter-spacing:-.035em;line-height:1.2;margin:0 0 24px}h3{font-size:18px;font-weight:600;margin:24px 0 10px}p{margin:10px 0 20px;max-width:790px}.eyebrow{font:12px/1.5 ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}.lead{font-size:20px;max-width:760px}.split{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:52px;align-items:start}.margin{font-size:14px;color:var(--muted);border-left:2px solid var(--line);padding-left:22px}.margin strong{color:var(--ink);font-size:15px}.numbers{display:flex;gap:45px;flex-wrap:wrap;margin:30px 0 44px}.numbers span{font-size:36px;line-height:1.1;letter-spacing:-.03em;display:block}.numbers small{font-size:13px;color:var(--muted)}.code{padding:28px 32px;background:#f4f7f9;border-left:3px solid var(--blue);font:clamp(14px,2vw,22px)/1.7 ui-monospace,Consolas,monospace;overflow:auto}.code .tail{color:var(--orange)}.note,figcaption{font-size:13px;line-height:1.6;color:var(--muted)}figure{margin:32px 0 0}img,video{display:block;width:100%;height:auto}video{background:#f9fbfc;border:1px solid var(--line);border-radius:2px}figcaption{margin:12px 0 0;max-width:910px}table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}td,th{text-align:left;border-bottom:1px solid var(--line);padding:14px 13px}thead th{font-weight:500;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}tbody th{font-weight:500}td small{display:block;color:var(--muted);font-size:12px;margin-top:4px}code{font-family:ui-monospace,Consolas,monospace;font-size:.88em}.formula{font-family:Georgia,serif;font-size:28px;line-height:1.7;padding:30px 0;color:var(--blue)}.links{display:flex;gap:22px;flex-wrap:wrap;font-size:14px;margin:24px 0}.check{display:grid;grid-template-columns:1fr 1fr;gap:0 36px}.check p{padding:20px 0;border-top:1px solid var(--line);font-size:14px}.check strong{font-size:17px;display:block;margin-bottom:7px}footer{max-width:1180px;padding:35px 28px 70px;margin:auto;color:var(--muted);font-size:12px}@media(max-width:800px){.split{grid-template-columns:1fr;gap:24px}.margin{max-width:600px}.check{grid-template-columns:1fr}section{padding:55px 0}h2{font-size:30px}.numbers{gap:25px}table{font-size:12px}td,th{padding:11px 6px}nav{gap:21px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body>
<nav aria-label="Report sections"><a href="#overview">Code</a><a href="#trellis">Trellis</a><a href="#energy">Energy</a><a href="#errors">Bit errors</a><a href="#uncertainty">Frame errors</a><a href="#validation">Validation</a></nav>
<main>
<section id="overview"><div class="eyebrow">Coding theory / synthetic BPSK over AWGN</div><h1>Convolutional coding</h1><p class="lead">A four-state encoder and source-built Viterbi decoder, measured with hard decisions and unquantized soft samples.</p>
<div class="numbers"><div><span>K = 3</span><small>constraint length</small></div><div><span>(7, 5)<sub style="font-size:18px">8</sub></span><small>generator polynomials</small></div><div><span>@@FRAMES@@</span><small>simulated frames</small></div><div><span>6,000</span><small>measured decoder calls</small></div></div>
<div class="split"><div><div class="eyebrow">Known hand vector</div><div class="code">input&nbsp;&nbsp; 1 0 1 1 <span class="tail">0 0</span><br>output&nbsp;11 10 00 01 <span class="tail">01 11</span><br>state&nbsp; 00 → 10 → 01 → 10 → 11 → 01 → 00</div><p class="note">Orange inputs terminate the frame. Output order is generator 7, then generator 5. State records the two previous inputs.</p></div><aside class="margin"><strong>Model</strong><p>Independent real Gaussian noise. Perfect carrier, timing and frame alignment. Equal-energy BPSK symbols.</p><strong>Implementation</strong><p>Python standard library encoder and decoder. No communications library or external decoder.</p></aside></div>
<div class="links"><a href="../README.md">Repository guide</a><a href="../results/published/frames.csv">Per-frame results</a><a href="../results/published/summary.csv">Aggregate CSV</a></div></section>
<section id="trellis"><div class="split"><div><h2>Viterbi traceback</h2><p>The replay draws saved survivor pointers and cumulative squared-distance metrics. It then traces backward from the required final state 00. Orange shows the hard-decoded path from the same noisy samples.</p></div><aside class="margin"><strong>Selected illustration</strong><p>Eight information bits at 2 dB, seed @@DEMO_SEED@@. This frame was selected for hard errors and successful soft recovery; it is excluded from the Monte Carlo estimates.</p></aside></div><figure><video controls preload="metadata" poster="assets/trellis.png" aria-label="Eight-second replay of saved Viterbi survivor paths"><source src="assets/trellis-replay.mp4" type="video/mp4"></video><figcaption>Eight seconds, silent, no autoplay. Playback time is unrelated to decoder runtime. <a href="../results/published/demo.json">Exact samples, metrics and predecessor pointers</a> · <a href="assets/trellis-replay.json">Replay provenance</a></figcaption></figure><figure><img src="assets/received-samples.png" alt="Actual received amplitudes for the selected 20-symbol frame; sign errors are orange" loading="lazy"><figcaption>Hard decoding keeps only each sign. Soft decoding retains distance from both possible transmitted levels, so a small wrong-sign sample has less influence than a confident observation.</figcaption></figure></section>
<section id="energy"><div class="split"><div><h2>Information-bit energy</h2><p>Every transmitted symbol consumes energy, including the four symbols generated by the two tail inputs. For a 64-bit payload, the effective rate is 64 / 132 = 0.484848, below the nominal rate of 1/2.</p><div class="formula">R<sub>eff</sub> = L / [2(L + 2)]<br>E<sub>b,info</sub> = E<sub>s</sub> / R<sub>eff</sub><br>σ² = 1 / (2R<sub>eff</sub> · 10<sup>Eb/N0[dB]/10</sup>)</div></div><aside class="margin"><strong>Normalization</strong><p>E<sub>s</sub> = 1 and real AWGN variance = N<sub>0</sub>/2. Coded and uncoded runs share information-bit Eb/N0, so their noise variances differ.</p><strong>Metric denominator</strong><p>Post-decoding BER counts 64 information bits. Pre-decoding channel errors count all 132 coded symbols, including tails.</p></aside></div><table><thead><tr><th>Per frame</th><th>Uncoded</th><th>Coded</th></tr></thead><tbody><tr><th>Information bits</th><td>64</td><td>64</td></tr><tr><th>Transmitted symbols</th><td>64</td><td>132</td></tr><tr><th>Information-bit energy</th><td>1</td><td>2.0625</td></tr><tr><th>Noise standard deviation at 3 dB</th><td>0.5006</td><td>0.7189</td></tr></tbody></table><p class="note">Values follow the declared unit-symbol-energy model; this is not a fixed-bandwidth or fixed-throughput comparison.</p></section>
<section id="errors"><div class="split"><div><h2>Information-bit error rates</h2><p>At 3 dB, soft decoding produced 68 errors; hard decoding produced 740; uncoded detection produced 581. Each method decoded 25,600 information bits at this point. The comparison includes termination energy.</p></div><aside class="margin"><strong>Observed tradeoff</strong><p>At 0 dB, both coded decoders had more information-bit errors than uncoded BPSK. The code does not improve every operating point.</p><strong>Finite experiment</strong><p>Zero observed soft errors at 6 dB is a count, not evidence of zero error probability.</p></aside></div><figure><img src="assets/information-ber.png" alt="Measured information-bit error curves comparing uncoded, hard Viterbi and soft Viterbi across five Eb/N0 points" loading="lazy"><figcaption>400 frames × 64 information bits per method and operating point. Dashed line: theoretical uncoded BPSK BER in ideal AWGN. No bitwise binomial intervals are used for the correlated decoded errors.</figcaption></figure><table><thead><tr><th>Eb(info)/N0</th><th>Uncoded errors / BER</th><th>Hard errors / BER</th><th>Soft errors / BER</th></tr></thead><tbody>@@ERROR_ROWS@@</tbody></table></section>
<section id="uncertainty"><div class="split"><div><h2>Frame-error uncertainty</h2><p>A frame fails if any of its information bits is wrong. Wilson 95% intervals apply to these equal-length frame outcomes under the independent, identically distributed AWGN simulation model at each fixed signal level.</p></div><aside class="margin"><strong>Paired coded observations</strong><p>Hard and soft decode the same noisy coded frame. Their estimates are paired; separate intervals are not a confidence interval for their difference.</p></aside></div><figure><img src="assets/block-errors.png" alt="Frame-error estimates and Wilson 95 percent intervals across five signal levels" loading="lazy"></figure><h3>At 6 dB</h3><table><thead><tr><th>Method</th><th>Failed frames</th><th>Wilson 95% interval</th></tr></thead><tbody>@@BLOCK_ROWS@@</tbody></table><p class="note">Soft decoding: 0 failures in 400 frames, with a model-based upper interval endpoint of 0.951%. This does not establish an information-bit BER bound.</p></section>
<section id="validation"><h2>Validation and reproducibility</h2><div class="check"><p><strong>Independent codeword enumeration</strong>1,344 short hard-observation sequences checked against exhaustive minimum-Hamming decoding. Another 120 soft frames checked against independently computed codeword distances.</p><p><strong>Termination and confidence</strong>Known hand vector, independent polynomial encoding, an otherwise cheaper illegal final state, weak sign errors, full traceback metrics and deterministic ties.</p><p><strong>Saved channel evidence</strong>Actual received samples stored as little-endian float64, with per-frame byte offsets and hashes. Inputs, decoded bits, state paths and all aggregate counts reconcile.</p><p><strong>Measured runtime</strong>Randomized method order and one warmup. Timings include complete decoder calls; channel generation and file I/O are excluded. Each call uses a different frame.</p></div><h3>Decoder timing at 3 dB</h3><table><thead><tr><th>Method</th><th>Median over 400 calls</th><th>Observed range</th></tr></thead><tbody>@@TIMING_ROWS@@</tbody></table><p class="note">@@OS@@ / Python @@PYTHON@@. These short-run measurements describe this machine and implementation, not modem throughput or real-time capability.</p><div class="links"><a href="../docs/algorithms.md">Algorithms and equations</a><a href="../docs/methodology.md">Simulation design</a><a href="../docs/findings.md">Findings and limits</a><a href="../results/published/environment.json">Run environment</a><a href="../results/published/manifest.json">Artifact hashes</a></div><p class="note">Primary references: <a href="https://doi.org/10.1109/TIT.1967.1054010">Viterbi (1967)</a>; <a href="https://web.mit.edu/6.02/www/currentsemester/handouts/L07_notes_pictures.pdf">MIT 6.02 Viterbi notes</a>. The code and all channel data in this repository were created with AI assistance. All signals are synthetic; no radio, modem or physical channel was measured.</p></section>
</main><footer>Convolutional coding lab · Terminated (7,5) code · Source, raw observations and reproducible evidence</footer></body></html>
"""
    replacements = {
        "FRAMES": f"{environment['total_frames']:,}",
        "DEMO_SEED": str(demo["seed"]),
        "ERROR_ROWS": "".join(error_rows),
        "BLOCK_ROWS": "".join(block_rows),
        "TIMING_ROWS": "".join(timing_rows),
        "OS": html.escape(environment["os"]),
        "PYTHON": html.escape(environment["python"]),
    }
    for key, value in replacements.items():
        content = content.replace("@@" + key + "@@", value)
    path.write_text(content, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/published")
    parser.add_argument("--out", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    assets = args.out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    summary = csv_rows(args.input / "summary.csv")
    environment = json.loads(
        (args.input / "environment.json").read_text(encoding="utf-8")
    )
    demo = json.loads((args.input / "demo.json").read_text(encoding="utf-8"))
    # The report prose describes the preserved study, not arbitrary pilot runs.
    if (
        environment["frames_per_point"] != 400
        or environment["information_length"] != 64
    ):
        raise ValueError(
            "The published report requires the declared 400-frame, 64-bit design"
        )
    plots(assets, summary, demo)
    write_report(args.out / "report.html", summary, environment, demo)
    files = [
        assets / name
        for name in ["information-ber.png", "block-errors.png", "received-samples.png"]
    ] + [args.out / "report.html"]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "source_sha256_lf": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        ).hexdigest(),
        "inputs": {
            f"results/published/{name}": hashlib.sha256(
                (args.input / name).read_bytes()
            ).hexdigest()
            for name in ["summary.csv", "environment.json", "demo.json"]
        },
        "outputs": {
            p.relative_to(args.out)
            .as_posix(): hashlib.sha256(p.read_bytes())
            .hexdigest()
            for p in files
        },
    }
    (assets / "figures.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"report": str(args.out / "report.html"), "figures": 3}))


if __name__ == "__main__":
    main()
