from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="config.yaml",
            imgsz = 640,
            epochs = 1000,
            batch = 5,
            patience = 0,
            optimizer = "SGD")

# results = model.predict("cat_dog.jpg")
# result = results[0]

# for box in result.boxes:
#     class_id = result.names[box.cls[0].item()]
#     cords = box.xyxy[0].tolist()
#     cords = [round(x) for x in cords]
#     conf = round(box.conf[0].item(), 2)
#     print("Object type:", class_id)
#     print("Coordinates:", cords)
#     print("Probability:", conf)
#     print("---")