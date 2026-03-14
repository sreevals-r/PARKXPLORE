"""
ParkXplore – Manual Parking Slot Marker (Point-Based)
Click on the image to place a point inside each parking slot.
Points are saved to a JSON file for use with check_occupancy.py.

Controls:
  • Left click     → place a slot point
  • 'u'            → undo last point
  • 's'            → save to JSON and quit
  • 'q' / ESC      → quit without saving
"""
import os
os.environ["XDG_SESSION_TYPE"] = "x11"

import cv2
import json
import argparse

# ── State ────────────────────────────────────────────────────────────
slots = []          # list of {"id": int, "x": int, "y": int}
temp_frame = None

POINT_COLOR = (0, 255, 255)   # yellow
POINT_RADIUS = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


def redraw(base_img):
    """Redraw all saved slot points on a fresh copy of the base image."""
    img = base_img.copy()
    for s in slots:
        cv2.circle(img, (s["x"], s["y"]), POINT_RADIUS, POINT_COLOR, -1)
        cv2.circle(img, (s["x"], s["y"]), POINT_RADIUS + 2, (0, 0, 0), 2)
        label = f'Slot {s["id"]}'
        cv2.putText(img, label, (s["x"] + 12, s["y"] + 5),
                    FONT, 0.5, (0, 255, 255), 1)
    return img


def mouse_cb(event, x, y, flags, param):
    global temp_frame
    base_img = param

    if event == cv2.EVENT_LBUTTONDOWN:
        slot = {"id": len(slots) + 1, "x": x, "y": y}
        slots.append(slot)
        print(f"  + Slot {slot['id']}: ({x}, {y})")
        temp_frame = redraw(base_img)


def main():
    global temp_frame

    parser = argparse.ArgumentParser(description="ParkXplore – Mark parking slots (point-based)")
    parser.add_argument("--image", default="dataset/test/images/daylight_part10_0035s_jpg.rf.eeae51087e66b58ca063124356a9e000.jpg",
                        help="Path to parking lot image")
    parser.add_argument("--output", default="parking_slots.json",
                        help="Output JSON file (default: parking_slots.json)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"❌  Image not found: {args.image}")
        return

    base_img = cv2.imread(args.image)
    if base_img is None:
        print(f"❌  Could not read image: {args.image}")
        return

    print("=" * 55)
    print("  ParkXplore – Parking Slot Marker (Point-Based)")
    print("=" * 55)
    print(f"  Image  : {args.image}")
    print(f"  Output : {args.output}")
    print("  Controls:")
    print("    Left click  → place slot point")
    print("    'u'         → undo last point")
    print("    's'         → save & quit")
    print("    'q' / ESC   → quit without saving")
    print("=" * 55 + "\n")

    temp_frame = redraw(base_img)

    win = "ParkXplore - Mark Slots (click to place, s=save, q=quit)"
    cv2.imshow(win, temp_frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(win, mouse_cb, base_img)

    while True:
        cv2.imshow(win, temp_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            if slots:
                with open(args.output, 'w') as f:
                    json.dump(slots, f, indent=2)
                print(f"\n✅  Saved {len(slots)} slot points to {args.output}")
            else:
                print("\n⚠️  No slots to save.")
            break

        elif key == ord('u'):
            if slots:
                removed = slots.pop()
                for i, s in enumerate(slots):
                    s["id"] = i + 1
                print(f"  ↩ Undone slot {removed['id']}")
                temp_frame = redraw(base_img)
            else:
                print("  ⚠️  Nothing to undo")

        elif key == ord('q') or key == 27:  # q or ESC
            print("\n❌  Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
