import cv2
import os
import glob

# ==========================================
# UPDATE THESE PATHS TO MATCH YOUR FOLDERS
# Example: "D:/PE2/train/images" 
# ==========================================
image_dir = r"D:\PARKEXPLORE\dataset\train_augmented\images"  # <--- CHECK THIS
label_dir = r"D:\PE2\PARKEXPLORE\train_augmented\labels"  # <--- CHECK THIS
classes = ["car"]

def verify_labels():
    print(f"Checking for images in: {image_dir}")
    
    # Check for jpg, jpeg, and png
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    print(f"Found {len(image_files)} images.")

    if len(image_files) == 0:
        print("ERROR: No images found! Check your 'image_dir' path.")
        return

    for i, img_path in enumerate(image_files):
        print(f"Opening {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        h, w, _ = img.shape
        
        # Find corresponding label file
        # Check for both .txt matching the image name
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, basename + ".txt")
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                print(f"  - Found label file with {len(lines)} boxes.")
                for line in lines:
                    parts = list(map(float, line.split()))
                    cls = int(parts[0])
                    x, y, nw, nh = parts[1], parts[2], parts[3], parts[4]
                    
                    x1 = int((x - nw/2) * w)
                    y1 = int((y - nh/2) * h)
                    x2 = int((x + nw/2) * w)
                    y2 = int((y + nh/2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = classes[cls] if cls < len(classes) else str(cls)
                    cv2.putText(img, label_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("  - No label file found for this image.")
        
        # Resize huge images to fit screen if necessary
        if w > 1920 or h > 1080:
             img = cv2.resize(img, (1280, 720))

        cv2.imshow("Label Verification", img)
        print("Press any key to see next image, or 'q' to quit.")
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_labels()