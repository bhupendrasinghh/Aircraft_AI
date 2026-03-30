from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    model = YOLO('yolov8n.pt')  # or yolov8m.pt for more accuracy

    results = model.train(
    data=r"C:/Users/ASHIRWAD PRATAPSINGH/Downloads/Innovation Hangar v2.v1-aug.yolov9/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,        # was 8, double it if VRAM allows
    device='cuda',
    workers=4,       # increase parallel loading
    cache=True,      # 🧠 cache images to RAM
    name='airplane_defect_yolov8_fast'
)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
