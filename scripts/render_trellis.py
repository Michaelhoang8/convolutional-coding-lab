"""Replay saved Viterbi metrics and survivor pointers; do not rerun decoding."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS, SECONDS = 1280, 760, 30, 8
INK, MUTED, BLUE, ORANGE = "#102537", "#526778", "#1959d6", "#bf6027"
PAPER, GRID = "#f9fbfc", "#dce4e9"


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def bit_text(values):
    return "".join(str(value) for value in values)


def xy(column, state):
    return 110 + column * 107, 280 + state * 90


def draw_frame(demo, seconds):
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((48, 30), "Viterbi survivor paths", font=font(34, True), fill=INK)
    draw.text(
        (48, 78),
        "Saved soft metrics / 8 information bits + 2 zero-tail inputs / state 00 at both ends",
        font=font(19),
        fill=MUTED,
    )
    draw.line((48, 118, 1232, 118), fill=GRID, width=2)
    for x, label, values, color in [
        (48, "TRANSMITTED", demo["information"], INK),
        (460, "HARD DECODE", demo["hard"]["information"], ORANGE),
        (870, "SOFT DECODE", demo["soft"]["information"], BLUE),
    ]:
        draw.text((x, 136), label, font=font(14, True), fill=MUTED)
        draw.text((x, 159), bit_text(values), font=font(30, True), fill=color)
    visible = min(10, max(0, int((seconds - 0.5) / 0.3)))
    back = min(10, max(0, int((seconds - 3.5) / 0.3)))
    for column in range(11):
        x, _ = xy(column, 0)
        draw.line((x, 242, x, 584), fill="#edf1f4", width=1)
        label = "start" if column == 0 else str(column)
        if column in (9, 10):
            label += " tail"
        draw.text((x, 218), label, anchor="mm", font=font(16), fill=MUTED)
    for state in range(4):
        _, y = xy(0, state)
        draw.text((51, y), f"{state:02b}", anchor="mm", font=font(20, True), fill=INK)
    for column in range(1, visible + 1):
        for state, predecessor in enumerate(
            demo["soft"]["predecessor_history"][column - 1]
        ):
            if predecessor is not None:
                draw.line(
                    (*xy(column - 1, predecessor), *xy(column, state)),
                    fill="#b9c8d4",
                    width=2,
                )
    if seconds >= 6.5:
        for column in range(1, 11):
            draw.line(
                (
                    *xy(column - 1, demo["hard"]["states"][column - 1]),
                    *xy(column, demo["hard"]["states"][column]),
                ),
                fill=ORANGE,
                width=4,
            )
    for column in range(11 - back, 11):
        draw.line(
            (
                *xy(column - 1, demo["soft"]["states"][column - 1]),
                *xy(column, demo["soft"]["states"][column]),
            ),
            fill=BLUE,
            width=6,
        )
    for column in range(11):
        for state in range(4):
            x, y = xy(column, state)
            value = demo["soft"]["metric_history"][column][state]
            active = column <= visible and value is not None
            on_path = (
                back > 0
                and column >= 10 - back
                and state == demo["soft"]["states"][column]
            )
            draw.ellipse(
                (x - 13, y - 13, x + 13, y + 13),
                fill=BLUE if on_path else PAPER,
                outline=BLUE if on_path else GRID,
                width=2,
            )
            if active:
                draw.text(
                    (x, y + 29), f"{value:.1f}", anchor="mm", font=font(14), fill=INK
                )
    if seconds < 0.5:
        phase = "Initial metric: state 00 = 0; other states unreachable."
    elif seconds < 3.5:
        phase = f"Forward pass: {visible} of 10 received pairs processed; one survivor per reachable state."
    elif seconds < 6.5:
        phase = (
            f"Traceback: {back} of 10 branches recovered from the known final state 00."
        )
    else:
        phase = "Blue: soft traceback. Orange: hard traceback from the same received samples."
    draw.text((48, 634), phase, font=font(21, True), fill=INK)
    draw.text(
        (48, 681),
        "Numbers are saved cumulative squared-distance metrics. Animation time is not decoder runtime.",
        font=font(17),
        fill=MUTED,
    )
    draw.text(
        (48, 712),
        f"Illustrative frame: seed {demo['seed']}, Eb(info)/N0 = 2 dB. Selected example; excluded from BER estimates.",
        font=font(16),
        fill=MUTED,
    )
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/published/demo.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    demo = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.out / "trellis-replay.mp4"
    poster = args.out / "trellis.png"
    draw_frame(demo, SECONDS).save(poster)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(FPS * SECONDS):
            process.stdin.write(draw_frame(demo, index / FPS).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pillow": pillow_version,
        "ffmpeg": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "input": "results/published/demo.json",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_sha256_lf": hashlib.sha256(source).hexdigest(),
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "poster_sha256": hashlib.sha256(poster.read_bytes()).hexdigest(),
        "duration_seconds": SECONDS,
        "fps": FPS,
        "frames": FPS * SECONDS,
        "dimensions": [WIDTH, HEIGHT],
        "audio": False,
        "scope": "Saved actual survivor pointers and metrics, then reverse traceback; animation timing is illustrative, not runtime. Hard path appears at the end for comparison.",
    }
    (args.out / "trellis-replay.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"video": str(output), "frames": FPS * SECONDS}))


if __name__ == "__main__":
    main()
