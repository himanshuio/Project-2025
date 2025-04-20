# trigger.py
import os
import cv2
from datetime import datetime

snapshot_folder = None  # Global but not set initially

def set_snapshot_folder(folder_path):
    global snapshot_folder
    snapshot_folder = folder_path
    os.makedirs(snapshot_folder, exist_ok=True) 
     
def trig(frame, trigger_type="threshold"):
    if snapshot_folder is None:
        raise ValueError("Snapshot folder not set. Use set_snapshot_folder() before calling trig().")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{trigger_type}_{timestamp}.jpg"
    filepath = os.path.join(snapshot_folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"[Snapshot Triggered] -> {filename}")
