from ultralytics import YOLO
from flask import request, Response, Flask, render_template
from waitress import serve
from PIL import Image
import json

app = Flask("Mosquito Detection", template_folder="./templates")
app.config.from_object("config.DevelopmentConfig")

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", 
        passes it through YOLOv8 object detection 
        network and returns an array of bounding boxes.
        :return: a JSON array of objects bounding 
        boxes in format 
        [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(json.dumps(boxes), mimetype='application/json')


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("../model/runs/detect/train2/weights/best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

# serve(app, host='0.0.0.0', port=8080)
if __name__ == "__main__":
    app.run(port = 8081)