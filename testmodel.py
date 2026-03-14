# D:/PE2/testmodel.py

from ultralytics import YOLO
import cv2
from pathlib import Path
import os

def test_yolo_model():
    """
    Test YOLOv11 model on test images and save detection results
    """

    # 1. Setup Paths
    model_path = 'D:/PARKEXPLORE/toy_car_finetune/3050_v12/weights/best.pt'
    test_images_path = 'D:/PARKEXPLORE/DA/test/images'
    output_dir = Path('D:/PARKEXPLORE/test_results2')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Model
    print("Loading YOLOv11 model...")
    model = YOLO(model_path)

    # 3. Get test images
    test_images = list(Path(test_images_path).glob('*.jpg'))

    if not test_images:
        print(f"No images found in {test_images_path}")
        return

    print(f"Found {len(test_images)} images. Running inference...\n")

    for img_path in test_images:
        print(f"Processing: {img_path.name}")

        # Run inference
        results = model.predict(img_path, conf=0.5, verbose=False)

        # Plot results (draw bounding boxes)
        img_result = results[0].plot()

        # Add model label
        cv2.putText(img_result, "YOLOv11 Detection",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

        # Save result image
        save_path = output_dir / f"yolo_test_{img_path.name}"
        cv2.imwrite(str(save_path), img_result)

    print("\n--- Testing Complete ---")
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    test_yolo_model()