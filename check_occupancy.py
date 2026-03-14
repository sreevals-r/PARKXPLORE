"""
ParkXplore – Parking Slot Occupancy Checker (Point-Based, Batch Mode)
Loads manually marked slot points from JSON, runs RT-DETR car detection
on ALL test images, and labels each slot as Occupied or Free.

Usage:
  python check_occupancy.py [--source <folder>] [--slots parking_slots.json]
"""
import os
os.environ["XDG_SESSION_TYPE"] = "x11"

import cv2
import json
import argparse
import glob
import numpy as np
from ultralytics import YOLO

# ── Config ───────────────────────────────────────────────────────────
MODEL_PATH = 'runs/parkxplore_yolo11/weights/best.pt'
DEFAULT_SLOTS = 'parking_slots.json'
SOURCE_PATH = 'dataset/test/images'
OUTPUT_DIR = 'output'
CONF = 0.4

# Colors (BGR)
COLOR_FREE     = (0, 200, 0)      # green
COLOR_OCCUPIED = (0, 0, 220)      # red
COLOR_CAR_BOX  = (255, 180, 0)    # cyan-ish
POINT_RADIUS   = 10

FONT = cv2.FONT_HERSHEY_SIMPLEX


def point_inside_box(px, py, box):
    """Check if point (px, py) is inside bounding box [x1, y1, x2, y2]."""
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def process_image(frame, slots, car_boxes):
    """Draw occupancy result on a single frame. Returns annotated frame and counts."""
    occupied_count = 0
    free_count = 0

    # Draw car bounding boxes
    for car in car_boxes:
        cv2.rectangle(frame, (car[0], car[1]), (car[2], car[3]),
                      COLOR_CAR_BOX, 2)
        clabel = f"Car {car[4]:.2f}"
        cv2.putText(frame, clabel, (car[0], car[1] - 5),
                    FONT, 0.45, COLOR_CAR_BOX, 1)

    # Check each slot point
    for slot in slots:
        px, py = slot["x"], slot["y"]
        is_occupied = any(point_inside_box(px, py, car) for car in car_boxes)

        if is_occupied:
            color = COLOR_OCCUPIED
            status = "Occupied"
            occupied_count += 1
        else:
            color = COLOR_FREE
            status = "Free"
            free_count += 1

        # Draw the slot point
        cv2.circle(frame, (px, py), POINT_RADIUS, color, -1)
        cv2.circle(frame, (px, py), POINT_RADIUS + 2, (0, 0, 0), 2)
        label = f"Slot {slot['id']}: {status}"
        cv2.putText(frame, label, (px + 15, py + 5), FONT, 0.5, color, 2)

    # Summary bar
    total = len(slots)
    summary = f"Total: {total}  |  Occupied: {occupied_count}  |  Free: {free_count}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(frame, summary, (10, 28), FONT, 0.7, (255, 255, 255), 2)

    return frame, occupied_count, free_count


def main():
    parser = argparse.ArgumentParser(description="ParkXplore – Batch occupancy checker (point-based)")
    parser.add_argument("--source", default=SOURCE_PATH, help="Folder of images to process")
    parser.add_argument("--slots",  default=DEFAULT_SLOTS, help="Path to parking_slots.json")
    parser.add_argument("--model",  default=MODEL_PATH, help="Path to RT-DETR weights")
    parser.add_argument("--conf",   type=float, default=CONF, help="Detection confidence")
    args = parser.parse_args()

    # ── Validate ─────────────────────────────────────────────────────
    if not os.path.isfile(args.slots):
        print(f"❌  Slots file not found: {args.slots}")
        print("    Run 'python mark_slots.py' first to mark slots.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load slots ───────────────────────────────────────────────────
    with open(args.slots) as f:
        slots = json.load(f)

    # ── Find images ──────────────────────────────────────────────────
    images = sorted(glob.glob(os.path.join(args.source, '*.jpg')) +
                    glob.glob(os.path.join(args.source, '*.png')))

    if not images:
        print(f"❌  No images found in {args.source}")
        return

    print("=" * 60)
    print("  ParkXplore – Batch Occupancy Checker (Point-Based)")
    print("=" * 60)
    print(f"  Source : {args.source}  ({len(images)} images)")
    print(f"  Slots  : {args.slots}  ({len(slots)} points)")
    print(f"  Model  : {args.model}")
    print(f"  Output : {OUTPUT_DIR}/")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────
    print("\nLoading model...")
    model = YOLO(args.model)
    print("Model loaded ✅\n")

    # ── Process all images ───────────────────────────────────────────
    total_occupied = 0
    total_free = 0

    for idx, img_file in enumerate(images):
        frame = cv2.imread(img_file)
        if frame is None:
            print(f"  [{idx+1:02d}/{len(images)}] ⚠️  Could not read {os.path.basename(img_file)}")
            continue

        # Detect cars
        results = model.predict(frame, conf=args.conf, verbose=False)
        car_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf_score = float(box.conf[0])
            car_boxes.append((x1, y1, x2, y2, conf_score))

        # Process and annotate
        annotated, occ, free = process_image(frame, slots, car_boxes)

        # Save
        out_name = f"occupancy_{os.path.basename(img_file)}"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, annotated)

        total_occupied += occ
        total_free += free
        print(f"  [{idx+1:02d}/{len(images)}] {os.path.basename(img_file):50s}  "
              f"Cars={len(car_boxes)}  Occupied={occ}  Free={free}  → saved")

    print(f"\n{'=' * 60}")
    print(f"  ✅  Done! Processed {len(images)} images")
    print(f"  🅿️  Slots per image  : {len(slots)}")
    print(f"  🔴  Total occupied   : {total_occupied}  (across all images)")
    print(f"  🟢  Total free       : {total_free}  (across all images)")
    print(f"  📁  Results saved to : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
