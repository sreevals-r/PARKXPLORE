from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('D:/PARKEXPLORE/runs/parkxplore_yolo11/weights/best.pt')
    results = model.train(
        data='D:/PARKEXPLORE/DA/data.yaml',
        epochs=30,
        imgsz=640,
        batch=4,
        amp=True,
        device=0,
        project='toy_car_finetune',
        name='3050_v1'
    )
