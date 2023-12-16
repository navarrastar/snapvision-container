from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import time
import json
import sqlite3
from vidgear.gears import CamGear
from flask_cors import CORS

application = Flask(__name__)
CORS(application, supports_credentials=True, origins="https://*.ext-twitch.tv")

stream_url = "https://www.twitch.tv/marvelsnapvision"

options = {"STREAM_RESOLUTION": "720p"}

# YOLOv8 models
classification_model = YOLO('classify.pt')
detection_model = YOLO('detect.pt')

cam = CamGear(source=stream_url, stream_mode=True, logging=True, **options).start()

previous_frames = {}

def get_coords():
    previous_time = time.time()
    frame_data = {}
    while True:
        current_time = time.time()

        frame = cam.read()

        time.sleep(0.01)

        if current_time - previous_time < 1:
            continue

        print(current_time)
        previous_frames[current_time] = frame

        if len(previous_frames) > 5:
            del previous_frames[list(previous_frames.keys())[0]]

        frame_data = {'time': current_time}

        detection_results = detection_model.predict(frame, iou=0.1, conf=0.5)

        # Get bounding box coordinates of each card detected and add them to frame data
        for box in range(0, len(detection_results[0].boxes.xyxy - 1)):
            coords = detection_results[0].boxes.xyxy[box].tolist()
            x, y, w, h = coords
            x, y, w, h = round(x), round(y), round(w), round(h)
            frame_data[box] = [x, y, w, h]

        yield f"data: {json.dumps(frame_data)}\n\n"
        previous_time = time.time()

def get_class_data(class_name):
    conn = sqlite3.connect('cards.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cards WHERE name=?", (class_name,))
    data = cursor.fetchone()

    conn.close()

    if data:
        return {"name": data[0], "description": data[3], "image": data[4]}
    else: return None

@application.route('/stream')
def stream():
    return Response(get_coords(), mimetype='text/event-stream')

with open('class_names.json') as f:
    class_names = json.load(f)

@application.route('/classify_card', methods=['POST'])
def classify_card():
    if request.method != 'POST':
        return
    card = request.get_json()
    card = json.loads(card)
    for time in previous_frames:
        if card['time'] == time:
            frame = previous_frames[time]
            x, y, w, h = card['coordinates']
            cropped_card = frame[y:h, x:w]
            classification_results = classification_model(cropped_card)
            card_class = classification_results[0].probs.top1
            return json.dumps(get_class_data(class_names[f'{card_class}']))
    return('', 204)

if __name__ == '__main__':
    application.run(ssl_context=('cert.pem', 'key.pem'))