from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
app = Flask(__name__)

# Global variables for storing user input
video_path = ""
model_path = ""
threshold = 0  

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_path, model_path, threshold
    people_count = 0

    if request.method == 'POST':
        video_path = request.form.get('rtsp_url', '')
        model_path = request.form.get('model_path', '')
        threshold = int(request.form.get('threshold', 0))
        detection_type = request.form.get("detection_type")

        if not video_path or not model_path:
            return "Error: RTSP URL or Model Path is missing!", 400

        print(f"RTSP URL: {video_path}")
        print(f"Model Path: {model_path}")
        print(f"Threshold: {threshold}")
        print(f"Threshold: {detection_type}")

        return redirect(url_for('stream' if detection_type == 'crowd' else 'stream2')) # Redirect to video feed

    return render_template("index.html")

@app.route('/stream')
def stream():
    return render_template("stream.html")  # Ensure stream.html exists

@app.route('/stream2')
def stream2():
    return render_template("stream2.html") 


def generate_frames(video_path, model_path, threshold,people_count):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Failed to open video: {video_path}")
        return

    height, width = int(cap.get(4)), int(cap.get(3))

    # Define ROI Zone
    polygons = np.array([
        [0, 0],
        [width - 5, 0],
        [width - 5, height - 5],
        [0, height - 5]
    ])
    zones = sv.PolygonZone(polygon=polygons)
    box_annotator = sv.BoxAnnotator(thickness=1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        # Run YOLO detection
        results = model(frame, imgsz=1248)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Apply zone mask
        mask = zones.trigger(detections=detections)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5) & mask]

        # Count detected people
        people_count = len(detections)

        # Annotate frame
        frame = box_annotator.annotate(scene=frame, detections=detections)
        cv2.putText(frame, f"People detected: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if people_count > threshold:
            cv2.putText(frame, "WARNING: Exceeds threshold!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
def violence(video_path, model_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load YOLOv8 Model
    model = YOLO(model_path)  # ✅ FIX: Load Model from Path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run YOLOv8 model on frame
        results = model(frame)  # ✅ FIX: Call model properly

        # Get the detected objects
        detections = results[0].boxes
        annotated_frame = results[0].plot()

        # Check for violence (assuming class 0 is a person)
        violence_detected = any(detection.cls == 0 for detection in detections)

        if violence_detected:
            cv2.putText(annotated_frame, "VIOLENCE DETECTED", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    global video_path, model_path, threshold
    return Response(generate_frames(video_path, model_path, threshold, people_count=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    global video_path, model_path
    if not video_path or not model_path:
        return "Error: Video path or model path is missing!", 400
    return Response(violence(video_path, model_path), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(debug=True)
