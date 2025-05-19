from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
from datetime import datetime, timedelta
from trigger import trig, set_snapshot_folder
from collections import deque

app = Flask(__name__, static_folder='static')

# Global variables
video_path = ""
model_path = ""
threshold = 0

snapshot_folder = os.path.join(app.static_folder, 'snapshots')
set_snapshot_folder(snapshot_folder)

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_path, model_path, threshold

    if request.method == 'POST':
        video_path = request.form.get('rtsp_url', '')
        model_path = request.form.get('model_path', '')
        threshold = int(request.form.get('threshold', 0))
        detection_type = request.form.get("detection_type")

        if not video_path or not model_path:
            return "Missing RTSP URL or Model Path!", 400

        return redirect(url_for('stream' if detection_type == 'crowd' else 'stream2'))

    return render_template("index.html")

@app.route('/stream')
def stream():
    return render_template("stream.html")

@app.route('/stream2')
def stream2():
    return render_template("stream2.html")

# -------- Stream Handlers -------- #

def generate_frames(video_path, model_path, threshold):
    last_trigger_time = None
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video.")
        return

    height, width = int(cap.get(4)), int(cap.get(3))

    polygons = np.array([[0, 0], [width-5, 0], [width-5, height-5], [0, height-5]])
    zones = sv.PolygonZone(polygon=polygons)
    box_annotator = sv.BoxAnnotator(thickness=1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, imgsz=1248)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask = zones.trigger(detections=detections)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5) & mask]

        people_count = len(detections)
        frame = box_annotator.annotate(scene=frame, detections=detections)
        cv2.putText(frame, f"People detected: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = datetime.now()
        if people_count > threshold:
            if last_trigger_time is None or (current_time - last_trigger_time).total_seconds() >= 30:
                trig(frame, trigger_type="threshold")
                last_trigger_time = current_time
            cv2.putText(frame, "WARNING: Exceeds threshold!", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


def violence(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError('Error opening video file.')

    # Time-Window Logic
    time_window = 3  # seconds
    frame_threshold = 5
    detection_times = deque(maxlen=frame_threshold)
    last_trigger_time = datetime.min
    frame_counter = 0
    time_counter = 0
    restarting_flag = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes
        annotated_frame = results[0].plot()

        violence_detected = any(detection.cls == 0 for detection in detections)

        if violence_detected:
            now = datetime.now()
            
            # New Logic: If time window exceeded without completing 5 frames
            if detection_times and (now - detection_times[0]).total_seconds() > time_window:
                print("Restarting...")
                detection_times.clear()
                frame_counter = 0
                time_counter = 0
                restarting_flag = True

            detection_times.append(now)
            frame_counter += 1
            time_counter = (now - detection_times[0]).total_seconds()

        # Display Frame Number and Time Counter
        cv2.putText(annotated_frame, f'Frame: {frame_counter}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f'Time: {int(time_counter)}s', (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display "Restarting..." on the screen if the window restarts
        if restarting_flag:
            cv2.putText(annotated_frame, "Restarting...", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            restarting_flag = False  # Reset the flag

        # Check if 5 detections happened within the time window
        if len(detection_times) == frame_threshold and time_counter <= time_window:
            cv2.putText(annotated_frame, 'VIOLENCE DETECTED', (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            detection_times.clear()
            frame_counter = 0
            time_counter = 0

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
@app.route('/video_feed')
def video_feed():
    global video_path, model_path, threshold
    return Response(generate_frames(video_path, model_path, threshold),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    global video_path, model_path
    return Response(violence(video_path, model_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------- Snapshot API for Flutter -------- #

@app.route('/list_snapshots')
def list_snapshots():
    if not os.path.exists(snapshot_folder):
        return jsonify([])

    files = sorted(os.listdir(snapshot_folder))
    urls = [
        f"{request.host_url}static/snapshots/{f}".replace("///", "//")
        for f in files if f.endswith('.jpg')
    ]
    return jsonify(urls)
# -------- Start Flask Server -------- #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
