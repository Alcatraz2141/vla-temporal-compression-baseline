from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def _read_gate_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_video_frames(path: Path) -> list[Image.Image]:
    frames = []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).convert("RGB"))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return frames


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(255, 255, 255))
    draw.multiline_text((x, y), text, font=font, fill=(0, 0, 0), spacing=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a contact sheet of event-memory gate activations.")
    parser.add_argument("--gate-csv", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--steps", type=str, default="80,100,120,140,160,200,240")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=180)
    args = parser.parse_args()

    rows = [row for row in _read_gate_rows(args.gate_csv) if int(row["valid_count"]) > 0]
    frames = _read_video_frames(args.video)
    by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["policy_step"])].append(row)

    requested_steps = [int(part.strip()) for part in args.steps.split(",") if part.strip()]
    selected: list[dict[str, str]] = []
    for step in requested_steps:
        if step not in by_step:
            continue
        selected.extend(sorted(by_step[step], key=lambda row: float(row["gate_score"]), reverse=True)[: args.top_k])

    if not selected:
        raise RuntimeError("No gate rows selected; check --steps and --gate-csv.")

    thumb_w = int(args.thumb_width)
    thumb_h = thumb_w
    label_h = 72
    cols = args.top_k
    rows_count = (len(selected) + cols - 1) // cols
    title_h = 62
    sheet = Image.new("RGB", (cols * thumb_w, title_h + rows_count * (thumb_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(18)
    label_font = _font(12)
    _draw_label(
        draw,
        (8, 8),
        "Event-gated ACT memory debug: top gate-weighted older chunks\n"
        "Note: current model softly weights all memory chunks; this sheet shows strongest chunks, not hard top-k retention.",
        title_font,
    )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy_step",
                "rank_within_step",
                "chunk_id",
                "chunk_start_frame",
                "chunk_end_frame",
                "selected_frame",
                "gate_score",
                "pool_attention_selected",
            ],
        )
        writer.writeheader()
        per_step_rank: dict[int, int] = defaultdict(int)
        for index, row in enumerate(selected):
            step = int(row["policy_step"])
            per_step_rank[step] += 1
            selected_frame = int(row["selected_frame"])
            frame_index = min(max(selected_frame, 0), len(frames) - 1)
            thumb = frames[frame_index].resize((thumb_w, thumb_h))
            x = (index % cols) * thumb_w
            y = title_h + (index // cols) * (thumb_h + label_h)
            sheet.paste(thumb, (x, y))
            label = (
                f"policy step {step}, rank {per_step_rank[step]}\n"
                f"chunk {row['chunk_id']} frames {row['chunk_start_frame']}-{row['chunk_end_frame']}\n"
                f"selected frame {selected_frame}\n"
                f"gate {float(row['gate_score']):.6f}"
            )
            _draw_label(draw, (x + 4, y + thumb_h + 4), label, label_font)
            writer.writerow(
                {
                    "policy_step": row["policy_step"],
                    "rank_within_step": per_step_rank[step],
                    "chunk_id": row["chunk_id"],
                    "chunk_start_frame": row["chunk_start_frame"],
                    "chunk_end_frame": row["chunk_end_frame"],
                    "selected_frame": row["selected_frame"],
                    "gate_score": row["gate_score"],
                    "pool_attention_selected": row["pool_attention_selected"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)
    print(args.summary_csv)


if __name__ == "__main__":
    main()
