
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
from ultralytics.nn.tasks import DetectionModel
import torch.serialization
import numpy as np
import csv
import os

torch.serialization.add_safe_globals([DetectionModel])

app = Flask(__name__)
model = None

# --------------------
# Globals for persistent tracking
# --------------------
next_id = 0
tracked_potholes = {}   # track_id : [bbox, last_seen_frame]
iou_threshold = 0.3
max_lost_frames = 5     # keep a pothole alive for 5 frames without match

# --------------------
# Model loader
# --------------------
def get_model():
    global model
    if model is None:
        model = YOLO("best.pt")
    return model

# --------------------
# Pothole info
# --------------------
def pothole_info(area_px):
    area_sq_m = area_px / 10000
    if area_px < 3000:
        return "Small", (0, 255, 0), area_sq_m, "Low"
    elif area_px < 8000:
        return "Medium", (0, 165, 255), area_sq_m, "Medium"
    else:
        return "Large", (0, 0, 255), area_sq_m, "High"

def pothole_speed(area_px):
    size, _, _, _ = pothole_info(area_px)
    if size == "Small":
        return 50
    elif size == "Medium":
        return 30
    else:
        return 15

def pothole_distance(x1, x2, fps=30, speed_kmh=40):
    focal_length = 800
    real_width = 0.6
    box_width = x2 - x1
    if box_width == 0:
        return 0
    distance = (real_width * focal_length) / box_width
    distance -= (speed_kmh * 1000 / 3600) / fps
    if distance < 0:
        distance = 0
    return round(distance, 2)

def pothole_depth(area_px):
    if area_px < 3000:
        return 5
    elif area_px < 8000:
        return 12
    else:
        return 20

# --------------------
# IOU function
# --------------------
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2Area = (box2[2]-box2[0])*(box2[3]-box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

# --------------------
# Video generator
# --------------------
def generate_frames(video_path):
    global next_id, tracked_potholes
    cap = cv2.VideoCapture(video_path)
    model = get_model()

    log_file = "static/potholes_log.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
    csvfile = open(log_file, mode="w", newline="")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Pothole_ID", "Frame", "Size", "Depth(cm)", "Severity", "Distance(m)"])
    
    frame_num = 0
    unique_ids = set()
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        speeds = []

        results = model(frame, conf=0.4, imgsz=416)

        detections = []
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2])

        new_tracked = {}
        for det in detections:
            assigned_id = None
            for track_id, (tbox, last_seen) in tracked_potholes.items():
                if iou(det, tbox) > iou_threshold:
                    assigned_id = track_id
                    break
            if assigned_id is None:
                assigned_id = next_id
                next_id += 1
            new_tracked[assigned_id] = [det, frame_num]  # update bbox and last seen
            unique_ids.add(assigned_id)

        # Remove lost potholes
        tracked_potholes = {tid: val for tid, val in new_tracked.items() if frame_num - val[1] <= max_lost_frames}

        for track_id, (box, _) in tracked_potholes.items():
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            area_px = box_width * (y2 - y1)
            size, color, area_sq_m, severity = pothole_info(area_px)
            depth = pothole_depth(area_px)
            distance = pothole_distance(x1, x2, fps=fps)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            start_y = y2 + 20
            cv2.putText(frame, f"ID:{track_id} {size} | {severity} | Distance:{distance}m",
                        (x1+5, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Depth:{depth}cm  Size:{area_sq_m:.2f}",
                        (x1+5, start_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            speeds.append(pothole_speed(area_px))
            csvwriter.writerow([track_id, frame_num, size, depth, severity, distance])

        if speeds:
            cv2.putText(frame, f"Recommended Speed: {min(speeds)} km/h", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Total Unique Potholes: {len(unique_ids)}", (int(width/2)-200, height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    csvfile.close()
    cap.release()

# --------------------
# Flask route
# --------------------
@app.route("/", methods=["GET","POST"])
def index():
    output_image = None
    if request.method=="POST":
        file = request.files["file"]
        if file:
            ext = file.filename.split(".")[-1].lower()
            model = get_model()
            
            if ext in ["jpg","jpeg","png"]:
                tmp = "temp.jpg"
                file.save(tmp)
                img = cv2.imread(tmp)
                img = cv2.resize(img, (640,640))
                speeds = []
                results = model(img, conf=0.25, imgsz=640)
                pothole_count = 0
                for r in results:
                    for box in r.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        box_width = x2 - x1
                        area_px = box_width * (y2 - y1)
                        size, color, area_sq_m, severity = pothole_info(area_px)
                        depth = pothole_depth(area_px)
                        distance = pothole_distance(x1, x2)
                        
                        cv2.rectangle(img,(x1,y1),(x2,y2), color,2)
                        start_y = y2+20
                        cv2.putText(img,f"{size} | {severity} | Distance:{round(distance,2)} m",
                                    (x1+5,start_y), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                        cv2.putText(img,f"Depth:{depth}cm  Size:{area_sq_m:.2f}",
                                    (x1+5,start_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                        speeds.append(pothole_speed(area_px))
                        pothole_count += 1
                
                if speeds:
                    cv2.putText(img,f"Recommended Speed: {min(speeds)} km/h",(30,40),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                
                height, width = img.shape[:2]
                cv2.putText(img,f"Total Potholes: {pothole_count}",(int(width/2)-150,height-20),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
                
                cv2.imwrite("static/output.jpg", img)
                output_image="output.jpg"
            
            elif ext in ["mp4","avi","mov","mkv"]:
                tmp="temp_video.mp4"
                file.save(tmp)
                return Response(generate_frames(tmp),
                                mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return render_template("index.html", output_image=output_image)

if __name__=="__main__":
    app.run(debug=True)