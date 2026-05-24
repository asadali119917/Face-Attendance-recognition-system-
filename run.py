#!/usr/bin/env python3
"""
FaceVision AI — University Face Recognition System
Entry point launcher

Usage:
    python run.py                  # Launch GUI
    python run.py --demo           # Demo mode (no camera needed)
    python run.py --headless IMAGE # Process single image
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def launch_gui():
    """Launch the full GUI application."""
    from gui.main_app import main
    print("🚀 Launching FaceVision AI GUI...")
    main()


def demo_mode():
    """Demo: generate synthetic database entries + process test image."""
    import numpy as np
    import cv2
    from core.face_engine import FaceRecognitionEngine

    print("=== FaceVision AI — Demo Mode ===\n")
    print("Initializing engine…")
    engine = FaceRecognitionEngine()

    # Create a synthetic test image with a face-like shape
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (20, 30, 50)
    cv2.ellipse(test_img, (320, 200), (80, 100), 0, 0, 360, (180, 150, 120), -1)
    cv2.circle(test_img, (295, 180), 12, (50, 50, 80), -1)
    cv2.circle(test_img, (345, 180), 12, (50, 50, 80), -1)
    cv2.ellipse(test_img, (320, 220), (30, 20), 0, 0, 180, (140, 100, 100), -1)

    print("Running detection on synthetic image…")
    annotated, results = engine.recognize_frame(test_img, mode='detection')
    print(f"  → Detected {len(results)} person(s)")

    print("\nRunning segmentation…")
    annotated_seg, results_seg = engine.recognize_frame(test_img, mode='segmentation')
    print(f"  → Segmented {len(results_seg)} person(s)")

    print(f"\nDatabase: {engine.db.count()} enrolled persons")
    print("\n✅ Demo completed. Run 'python run.py' for the full GUI.\n")


def headless_image(image_path: str):
    """Process a single image without GUI."""
    import cv2
    from core.face_engine import FaceRecognitionEngine

    print(f"Processing: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("ERROR: Cannot read image")
        return

    engine = FaceRecognitionEngine()
    annotated, results = engine.recognize_frame(frame, 'detection')

    out_path = str(Path(image_path).parent / f"result_{Path(image_path).name}")
    cv2.imwrite(out_path, annotated)
    print(f"Output saved: {out_path}")
    print(f"Detected {len(results)} person(s):")
    for r in results:
        print(f"  • {r.name} | Conf: {r.confidence:.1%} | "
              f"Emotion: {r.emotion} | Liveness: {r.liveness_score:.2f}")


if __name__ == '__main__':
    if '--demo' in sys.argv:
        demo_mode()
    elif '--headless' in sys.argv:
        idx = sys.argv.index('--headless')
        if idx + 1 < len(sys.argv):
            headless_image(sys.argv[idx + 1])
        else:
            print("Usage: python run.py --headless <image_path>")
    else:
        launch_gui()
