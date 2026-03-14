from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os
import csv
import numpy as np

# Fix Ultralytics config warning
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

app = Flask(__name__)
model = None


# --------------------
# Load YOLO model
# --------------------
def get_model():
    global model
    if model is None:
        model = YOLO("best.pt")
    return model


# --------------------
# Pothole information
# --------------------
def pothole_info(area_px):
    area_sq_m = area_px / 10000
    if area_px < 3000:
        return "Small", (0,255,0), area_sq_m, "Low"
    elif area_px < 8000:
        return "Medium", (0,165,255), area_sq_m, "Medium"
    else:
        return "Large", (0,0,255), area_sq_m, "High"


def pothole_speed(area_px):
    size,_,_,_ = pothole_info(area_px)
    if size=="Small":
        return 50
    elif size=="Medium":
        return 30
    else:
        return 15


def pothole_depth(area_px):
    if area_px < 3000:
        return 5
    elif area_px < 8000:
        return 12
    else:
        return 20


def pothole_distance(x1,x2):
    focal_length = 800
    real_width = 0.6
    box_width = x2-x1
    if box_width==0:
        return 0
    distance=(real_width*focal_length)/box_width
    return round(distance,2)


# --------------------
# Image Detection
# --------------------
@app.route("/", methods=["GET","POST"])
def index():

    output_image=None

    if request.method=="POST":

        file = request.files.get("file")

        if not file:
            return "No file uploaded"

        ext=file.filename.split(".")[-1].lower()
        model=get_model()

        # ---------------- IMAGE ----------------
        if ext in ["jpg","jpeg","png"]:

            tmp="temp.jpg"
            file.save(tmp)

            img=cv2.imread(tmp)

            if img is None:
                return "Image read error"

            img=cv2.resize(img,(640,640))

            results=model(img,conf=0.4,imgsz=416)

            speeds=[]
            pothole_count=0

            for r in results:
                for box in r.boxes.xyxy:

                    x1,y1,x2,y2=map(int,box)

                    box_width=x2-x1
                    area_px=box_width*(y2-y1)

                    size,color,area_sq_m,severity=pothole_info(area_px)
                    depth=pothole_depth(area_px)
                    distance=pothole_distance(x1,x2)

                    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

                    start_y=y2+20

                    cv2.putText(img,f"{size} | {severity} | Dist:{distance}m",
                                (x1+5,start_y),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

                    cv2.putText(img,f"Depth:{depth}cm  Size:{area_sq_m:.2f}",
                                (x1+5,start_y+20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

                    speeds.append(pothole_speed(area_px))
                    pothole_count+=1

            if speeds:
                cv2.putText(img,f"Recommended Speed: {min(speeds)} km/h",
                            (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

            h,w=img.shape[:2]

            cv2.putText(img,f"Total Potholes: {pothole_count}",
                        (int(w/2)-150,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

            output_path=os.path.join("static","output.jpg")
            cv2.imwrite(output_path,img)

            output_image="output.jpg"


    return render_template("index.html", output_image=output_image)


if __name__=="__main__":
    port=int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)
