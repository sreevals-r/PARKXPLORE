# D:/PE2/preprocess_and_augment.py

import albumentations as A
import cv2
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

class EnhancedParkingDataProcessor:
    """
    FIXED: Enhanced preprocessing and augmentation pipeline for ParkXplore
    """
    
    def __init__(self, base_path="D:/PARKEXPLORE/dataset"):
        self.base_path = Path(base_path)
        self.train_img = self.base_path / "train" / "images"
        self.train_lbl = self.base_path / "train" / "labels"
        
        self.aug_train_img = self.base_path / "train_augmented" / "images"
        self.aug_train_lbl = self.base_path / "train_augmented" / "labels"
        
        os.makedirs(self.aug_train_img, exist_ok=True)
        os.makedirs(self.aug_train_lbl, exist_ok=True)
    
    def get_augmentation_pipeline(self, mode='enhanced'):
        """
        FIXED: Enhanced augmentation pipeline with corrected parameters
        """
        
        if mode == 'minimal':
            return A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.GaussNoise(var_limit=(10.0, 40.0), mean=0, p=0.4),  # FIXED
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        
        elif mode == 'standard':
            return A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomToneCurve(scale=0.3, p=1.0),
                ], p=0.5),
                A.GaussNoise(var_limit=(10.0, 40.0), mean=0, p=0.4),  # FIXED
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.Rotate(limit=5, p=0.2, border_mode=cv2.BORDER_CONSTANT),  # FIXED: removed 'value'
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        
        elif mode == 'enhanced':
            return A.Compose([
                # ============================================================
                # CRITICAL AUGMENTATIONS (Mandatory)
                # ============================================================
                
                # Lighting variations (Day/Night/Purple lighting)
                A.RandomBrightnessContrast(
                    brightness_limit=0.35,
                    contrast_limit=0.35,
                    p=0.7
                ),
                
                # Exposure/Gamma variations
                A.OneOf([
                    A.RandomGamma(gamma_limit=(75, 125), p=1.0),
                    A.RandomToneCurve(scale=0.4, p=1.0),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                ], p=0.5),
                
                # Camera noise (especially nighttime)
                A.GaussNoise(var_limit=(10.0, 45.0), mean=0, p=0.4),  # FIXED
                
                # Motion blur and focus issues
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=1.0),
                ], p=0.35),
                
                # ============================================================
                # HIGH PRIORITY AUGMENTATIONS
                # ============================================================
                
                # Color variations (Purple lighting + general color shifts)
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=25,
                    val_shift_limit=15,
                    p=0.4
                ),
                
                # Small rotation for camera angle robustness
                A.Rotate(
                    limit=7,
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT  # FIXED: removed 'value' parameter
                ),
                
                # ============================================================
                # MEDIUM PRIORITY AUGMENTATIONS
                # ============================================================
                
                # Horizontal flip (ONLY for top-down view)
                A.HorizontalFlip(p=0.15),
                
                # Slight random crop for edge robustness
                A.RandomResizedCrop(
                    size=(640, 640),  # FIXED: added required 'size' parameter
                    scale=(0.92, 1.0),
                    ratio=(0.95, 1.05),
                    p=0.2
                ),
                
                # Cutout/CoarseDropout for occlusion robustness
                A.CoarseDropout(
                    max_holes=2,
                    max_height=80,
                    max_width=80,
                    min_holes=1,
                    min_height=40,
                    min_width=40,
                    fill_value=0,
                    p=0.2
                ),
                
                # ============================================================
                # LOW PRIORITY AUGMENTATIONS
                # ============================================================
                
                # Random shadow (simulates lighting variations)
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    p=0.15
                ),
                
                # Random fog/haze (simulates weather)
                A.RandomFog(
                    fog_coef_range=(0.1, 0.3),
                    alpha_coef=0.1,
                    p=0.1
                ),
                
                # Slight perspective shift (camera angle variations)
                A.Perspective(
                    scale=(0.02, 0.05),
                    keep_size=True,
                    p=0.15
                ),
                
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.25,
                min_area=80
            ))
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'minimal', 'standard', or 'enhanced'")
    
    def preprocess_image(self, image):
        """
        Enhanced preprocessing with better lighting normalization
        """
        # CLAHE in LAB color space (for purple lighting)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return processed
    
    def load_yolo_annotations(self, label_path):
        """Load YOLO format annotations"""
        bboxes = []
        class_labels = []
        
        if not label_path.exists():
            return bboxes, class_labels
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    bboxes.append([x_c, y_c, w, h])
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def save_yolo_annotations(self, path, bboxes, labels):
        """Save YOLO format annotations"""
        with open(path, 'w') as f:
            for bbox, cls in zip(bboxes, labels):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")
    
    def process_dataset(self, augmentations_per_image=3, mode='enhanced', apply_preprocessing=True):
        """
        Enhanced processing pipeline
        """
        transform = self.get_augmentation_pipeline(mode)
        
        image_files = list(self.train_img.glob('*.jpg')) + \
                     list(self.train_img.glob('*.png')) + \
                     list(self.train_img.glob('*.jpeg'))
        
        print(f"\n{'='*70}")
        print(f"🚗 PARKXPLORE - ENHANCED DATA PREPROCESSING & AUGMENTATION 🚗")
        print(f"{'='*70}")
        print(f"📁 Source: {self.train_img}")
        print(f"📁 Output: {self.aug_train_img}")
        print(f"📊 Images found: {len(image_files)}")
        print(f"🎨 Augmentation mode: {mode.upper()}")
        print(f"🔢 Augmentations per image: {augmentations_per_image}")
        print(f"🖼️  Preprocessing (CLAHE): {'ENABLED' if apply_preprocessing else 'DISABLED'}")
        print(f"{'='*70}\n")
        
        if mode == 'enhanced':
            print("✨ ENHANCED MODE ACTIVE - Additional augmentations:")
            print("   ✅ Horizontal flip (15% probability)")
            print("   ✅ Gentle crops (20% probability)")
            print("   ✅ Cutout/occlusion simulation (20% probability)")
            print("   ✅ Random shadows (15% probability)")
            print("   ✅ Random fog (10% probability)")
            print("   ✅ Perspective shifts (15% probability)")
            print("   ✅ Enhanced color variations")
            print("   ✅ Defocus blur simulation\n")
        
        successful = 0
        failed = 0
        total_generated = 0
        
        for img_path in tqdm(image_files, desc=f"Processing ({mode} mode)"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"❌ Cannot read: {img_path.name}")
                    failed += 1
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if apply_preprocessing:
                    image = self.preprocess_image(image)
                
                label_path = self.train_lbl / f"{img_path.stem}.txt"
                bboxes, class_labels = self.load_yolo_annotations(label_path)
                
                if len(bboxes) == 0:
                    print(f"⚠️  No annotations: {img_path.name}")
                    continue
                
                # Save original
                orig_img = self.aug_train_img / img_path.name
                cv2.imwrite(str(orig_img), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                orig_lbl = self.aug_train_lbl / f"{img_path.stem}.txt"
                self.save_yolo_annotations(orig_lbl, bboxes, class_labels)
                total_generated += 1
                
                # Generate augmented versions
                aug_count = 0
                attempts = 0
                max_attempts = augmentations_per_image * 3
                
                while aug_count < augmentations_per_image and attempts < max_attempts:
                    attempts += 1
                    
                    try:
                        augmented = transform(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )
                        
                        aug_image = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_labels = augmented['class_labels']
                        
                        # Skip if too many boxes were removed
                        if len(aug_bboxes) < len(bboxes) * 0.5:
                            continue
                        
                        # Save augmented data
                        aug_name = f"{img_path.stem}_aug{aug_count}{img_path.suffix}"
                        aug_img_path = self.aug_train_img / aug_name
                        cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                        
                        aug_lbl_path = self.aug_train_lbl / f"{img_path.stem}_aug{aug_count}.txt"
                        self.save_yolo_annotations(aug_lbl_path, aug_bboxes, aug_labels)
                        
                        total_generated += 1
                        aug_count += 1
                        
                    except Exception as e:
                        continue
                
                successful += 1
                
            except Exception as e:
                print(f"❌ Error with {img_path.name}: {e}")
                failed += 1
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successfully processed: {successful}/{len(image_files)} images")
        print(f"❌ Failed: {failed} images")
        print(f"📊 Total images generated: {total_generated}")
        print(f"   └─ Original: {successful}")
        print(f"   └─ Augmented: {total_generated - successful}")
        print(f"📁 Saved to: {self.aug_train_img.parent}")
        print(f"{'='*70}\n")
        
        return successful, failed, total_generated
    
    def create_training_yaml(self):
        """Create optimized training configuration"""
        yaml_content = {
            'path': 'D:/PE2/dataset',
            'train': 'train_augmented/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['car']
        }
        
        yaml_path = self.base_path / "data_augmented.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Created training config: {yaml_path}\n")
        
        return yaml_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚗"*35)
    print("     PARKXPLORE - ENHANCED DATA PREPARATION PIPELINE (FIXED)")
    print("🚗"*35 + "\n")
    
    # Initialize processor
    processor = EnhancedParkingDataProcessor(base_path="D:/PE2/dataset")
    
    # User choice
    print("Select augmentation mode:")
    print("1. MINIMAL     - Only critical augmentations (fastest)")
    print("2. STANDARD    - Recommended baseline (balanced)")
    print("3. ENHANCED    - Maximum robustness (RECOMMENDED) ⭐")
    print()
    
    choice = input("Enter choice (1/2/3) [default=3]: ").strip() or "3"
    
    mode_map = {'1': 'minimal', '2': 'standard', '3': 'enhanced'}
    selected_mode = mode_map.get(choice, 'enhanced')
    
    aug_per_image = 3 if selected_mode != 'enhanced' else 4
    
    # Run processing
    successful, failed, total = processor.process_dataset(
        augmentations_per_image=aug_per_image,
        mode=selected_mode,
        apply_preprocessing=True
    )
    
    # Create training config
    if successful > 0:
        yaml_path = processor.create_training_yaml()
        
        print("="*70)
        print("🎯 NEXT STEPS:")
        print("="*70)
        print("1. ✅ Data preprocessing & augmentation complete")
        print("2. ⏭️  Train model: python train_model.py")
        print(f"3. 📝 Training config: {yaml_path}")
        print("="*70 + "\n")
