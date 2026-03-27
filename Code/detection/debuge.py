from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm
import math

# ================== PATH ==================
VIDEO_PATH = r"D:\KY_4\DAP\Dap391\Project\Sources\Video\video_unlabeled\GOVAP4K.mp4"
OUTPUT_DIR = r"D:\KY_4\DAP\Dap391\Project\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_VIDEO = os.path.join(OUTPUT_DIR, "final_violation_with_plate_crop.mp4")
VIOLATIONS_DIR = os.path.join(OUTPUT_DIR, "violations_with_plate_crop")

os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# ================== MODELS ==================
light_model = YOLO(
    r'D:\KY_4\DAP\Dap391\Project\Trained_model_main\Traffic_light_violationight_violation-main\Model_train\traffic_light_model_update_data(morning_afternoon_night)\weights\best.pt'
)
vehicle_model = YOLO(r'd:\KY_4\DAP\Dap391\Project\Code\yolo26s.pt')

#  PLATE DETECTION MODEL
plate_model = YOLO(r"D:\KY_4\DAP\Dap391\Project\LP_v26\model\plate_model\weights\best.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

LIGHT_CONF = 0.01
STOPLINE_CONF = 0.4

#  PLATE DETECTION CONFIG
PLATE_CONF = 0.5

RED_FLASH_DURATION = 10
RED_FLASH_ALPHA = 0.3

MOVEMENT_THRESHOLD = 3

# ================== CLAHE SETUP ==================
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# ================== UTILS ==================
def calculate_distance(box1, box2):
    c1x = (box1[0] + box1[2]) / 2
    c1y = (box1[1] + box1[3]) / 2
    c2x = (box2[0] + box2[2]) / 2
    c2y = (box2[1] + box2[3]) / 2
    return math.sqrt((c2x - c1x) ** 2 + (c2y - c1y) ** 2)


def get_light_and_stopline(frame):
    results = light_model.predict(frame, conf=0.1, imgsz=640, verbose=False)[0]

    traffic_state = "UNKNOWN"
    best_light_conf = 0.0
    best_stopline = None
    best_stopline_conf = 0.0

    if results.boxes is None:
        return traffic_state, []

    for box, cls_id, conf in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.cls.cpu().numpy(),
        results.boxes.conf.cpu().numpy()
    ):
        name = light_model.names[int(cls_id)].lower()

        if "red" in name and conf >= LIGHT_CONF and conf > best_light_conf:
            best_light_conf = conf
            traffic_state = "RED"
        elif "yellow" in name and conf >= LIGHT_CONF and conf > best_light_conf:
            best_light_conf = conf
            traffic_state = "YELLOW"
        elif "green" in name and conf >= LIGHT_CONF and conf > best_light_conf:
            best_light_conf = conf
            traffic_state = "GREEN"
        elif "stop" in name and conf >= STOPLINE_CONF and conf > best_stopline_conf:
            best_stopline_conf = conf
            best_stopline = list(map(int, box))

    stoplines = [best_stopline] if best_stopline else []
    return traffic_state, stoplines


#  FUNCTION: DETECT PLATE
def detect_license_plate(vehicle_crop):
    """
    Detect license plate từ vehicle crop
    Input: vehicle crop (BGR)
    Output: plate info dict hoặc None
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None
    
    try:
        results = plate_model.predict(vehicle_crop, conf=PLATE_CONF, imgsz=640, verbose=False)[0]
        
        if results.boxes is None or len(results.boxes) == 0:
            return None
        
        # Lấy plate có diện tích lớn nhất
        plates = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        best_idx = np.argmax([(b[2]-b[0]) * (b[3]-b[1]) for b in plates])
        best_plate = plates[best_idx]
        best_conf = confs[best_idx]
        
        return {
            "box": list(map(int, best_plate)),
            "conf": float(best_conf)
        }
    except Exception as e:
        print(f"  Plate detection error: {e}")
        return None


#  FUNCTION: CROP PLATE (NO CLAHE - raw crop only)
def crop_plate_raw(vehicle_crop, plate_info):
    """
    Crop biển số từ vehicle crop (KHÔNG áp dụng CLAHE)
    Input: vehicle crop, plate detection result
    Output: plate crop (raw, chưa enhance)
    """
    if plate_info is None:
        return None
    
    x1, y1, x2, y2 = plate_info["box"]
    
    # Thêm padding nhỏ
    pad = 5
    h, w = vehicle_crop.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    # Crop plate (raw - không CLAHE)
    plate_crop = vehicle_crop[y1:y2, x1:x2]
    
    if plate_crop.size == 0:
        return None
    
    return plate_crop


#  FUNCTION: APPLY CLAHE ON PLATE
def apply_clahe_on_plate(plate_crop):
    """
    Áp dụng CLAHE 1 lần trên plate crop
    Input: plate crop (BGR)
    Output: plate crop sau CLAHE (BGR)
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    
    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    plate_clahe = clahe.apply(plate_gray)
    plate_enhanced = cv2.cvtColor(plate_clahe, cv2.COLOR_GRAY2BGR)
    
    return plate_enhanced


def smart_crop_vehicle(frame, x1, y1, x2, y2, padding_ratio=0.2):
    """Smart crop vehicle with adaptive padding"""
    h, w = frame.shape[:2]
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)
    
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)
    
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def smart_crop_license_plate(vehicle_crop, vehicle_bbox=None):
    """
    Detect and crop license plate region from vehicle crop (RAW - không CLAHE)
    Dùng model plate detection để tìm biển số
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, False
    
    plate_info = detect_license_plate(vehicle_crop)
    
    if plate_info is None:
        # Fallback: crop region dưới xe (nếu không detect được)
        h, crop_w = vehicle_crop.shape[:2]
        plate_top = int(h * 0.55)
        plate_bottom = int(h * 0.95)
        plate_left = int(crop_w * 0.05)
        plate_right = int(crop_w * 0.95)
        
        plate_region = vehicle_crop[plate_top:plate_bottom, plate_left:plate_right]
        return plate_region, False
    
    # Crop biển số (RAW - không CLAHE)
    plate_raw = crop_plate_raw(vehicle_crop, plate_info)
    
    if plate_raw is None:
        return None, False
    
    return plate_raw, True


# ================== VIDEO IO ==================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

SIDEBAR_WIDTH = 600
SIDEBAR_HEIGHT = h

OUTPUT_WIDTH = w + SIDEBAR_WIDTH
OUTPUT_HEIGHT = h 

out = cv2.VideoWriter(
    OUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (OUTPUT_WIDTH, OUTPUT_HEIGHT)
)

print(f"Output size: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
print(f"Original video: {w}x{h}, Sidebar: {SIDEBAR_WIDTH}x{h}")
print(f"Plate detection: ENABLED with model")

# ================== MEMORY ==================
vehicle_memory = {}
cached_stoplines = []
red_flash_remaining = 0
red_start_frame = None
violation_counter = 0
MAX_IDLE_FRAMES = int(15 * fps)  # 15 giây video
violation_history = []
current_display_crop = None
last_violation_id = None

print("START FINAL RUN - WITH PLATE DETECTION")
print(f"Total frames: {total_frames}")
print(f"Output directory: {OUTPUT_DIR}")
print("-" * 50)

# ================== MAIN LOOP ==================
pbar = tqdm(total=total_frames, desc="Processing", unit="frame", ncols=100)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    current_frame_crop = None
    
    # -------- Traffic light & stopline --------
    traffic_state, stoplines = get_light_and_stopline(frame)
    if stoplines:
        cached_stoplines = stoplines
    current_stoplines = stoplines if stoplines else cached_stoplines

    # -------- RED TIMER --------
    if traffic_state == "RED":
        if red_start_frame is None:
            red_start_frame = frame_count
    else:
        red_start_frame = None

    red_elapsed_time = 0
    if red_start_frame is not None:
        red_elapsed_time = (frame_count - red_start_frame) / fps

    # -------- Vehicle tracking --------
    results = vehicle_model.track(
        frame,
        persist=True,
        conf=0.35,
        iou=0.3,
        imgsz=1280,
        tracker="botsort.yaml",
        device=0,
        verbose=False
    )[0]

    if results.boxes is not None and results.boxes.id is not None:
        for box, tid, cls_id in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.id.cpu().numpy(),
            results.boxes.cls.cpu().numpy()
        ):
            if int(cls_id) not in VEHICLE_CLASSES:
                continue

            tid = int(tid)
            veh_box = box.tolist()
            x1, y1, x2, y2 = map(int, veh_box)

            # rear point
            rear_x = (x1 + x2) // 2
            rear_y = y2

            state = vehicle_memory.setdefault(
                tid,
                {
                    "violated": False,
                    "saved": False,
                    "prev_box": None,
                    "prev_rear_y": None,
                    "prev_rear_x": None,
                    "already_crossed": False,
                    "crossed_vehical": set()
                }
            )

            # ================== CROSSING LOGIC ==================
            crossed = False
            if not state.get("already_crossed", False):
                if state["prev_rear_y"] is not None:
                    prev_y = state["prev_rear_y"]
                    curr_y = rear_y
                    prev_x = state.get("prev_rear_x", rear_x)
                    curr_x = rear_x
                    vehical_id = tid
                
                    if vehical_id not in state["crossed_vehical"]:
                        for sl in current_stoplines:
                            sx1, sy1, sx2, sy2 = sl

                            stopline_center = (sy1 + sy2) / 2
                            stopline_top = sy1

                            # VI PHẠM: Chấm đỏ (rear_y) nằm trong vùng [top, center)
                            in_violation_zone = (stopline_top <= rear_y < stopline_center)
                            rear_x_in_range = (sx1 <= rear_x <= sx2)

                            if in_violation_zone and rear_x_in_range:
                                crossed = True
                                state["crossed_vehical"].add(vehical_id)
                                state["already_crossed"] = True
                                break

            state["prev_rear_y"] = rear_y
            state["prev_rear_x"] = rear_x

            # ---- FINAL VIOLATION ----
            if (
                traffic_state in ["RED", "YELLOW"]
                and crossed
                and not state["violated"]
            ):
                state["violated"] = True
                red_flash_remaining = RED_FLASH_DURATION
            
            # ========== SAVE VIOLATION + PLATE CROP ==========
            if state["violated"] and not state["saved"]:
                violation_counter += 1
                
                violation_folder = os.path.join(
                    VIOLATIONS_DIR, 
                    f"violation_{violation_counter:05d}_ID{tid}_frame{frame_count}"
                )
                os.makedirs(violation_folder, exist_ok=True)
                
                # ═══ CROP VEHICLE (KHÔNG CLAHE) ═══
                pad = 10
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                
                # Lưu vehicle crop RAW (không CLAHE)
                crop_path = os.path.join(violation_folder, "01_crop_vehicle.jpg")
                cv2.imwrite(crop_path, crop)
                
                # ═══ DETECT PLATE & CROP (RAW) ═══
                plate_crop, plate_detected = smart_crop_license_plate(crop)
                plate_conf = 0.0
                
                if plate_crop is not None:
                    # Lấy confidence của plate
                    plate_info = detect_license_plate(crop)
                    if plate_info:
                        plate_conf = plate_info["conf"]
                    
                    # ═══ CLAHE LẦN 1 trên plate crop RAW ═══
                    plate_clahe_1 = apply_clahe_on_plate(plate_crop)
                    if plate_clahe_1 is not None:
                        plate_path_1 = os.path.join(violation_folder, "02_crop_plate_clahe_1.jpg")
                        cv2.imwrite(plate_path_1, plate_clahe_1)
                        print(f"  ✓ Plate CLAHE lần 1 saved")
                    else:
                        print(f"  ✗ Plate CLAHE lần 1 failed — skipping CLAHE lần 2")
                    
                    # ═══ CLAHE LẦN 2 trên plate đã CLAHE lần 1 ═══
                    plate_clahe_2 = apply_clahe_on_plate(plate_clahe_1) if plate_clahe_1 is not None else None
                    if plate_clahe_2 is not None:
                        plate_path_2 = os.path.join(violation_folder, "03_crop_plate_clahe_2.jpg")
                        cv2.imwrite(plate_path_2, plate_clahe_2)
                        print(f"  ✓ Plate CLAHE lần 2 saved")
                    else:
                        print(f"  ✗ Plate CLAHE lần 2 failed — no output saved")
                    
                    print(f"  ✓ License plate detected and saved (conf: {plate_conf:.2f})")
                else:
                    print(f"    No license plate detected (using fallback)")
                
                # ═══ SAVE FULL FRAME ═══
                full_path = os.path.join(violation_folder, "04_full_frame_annotated.jpg")
                cv2.imwrite(full_path, annotated)
                
                # ═══ SAVE METADATA ═══
                metadata_path = os.path.join(violation_folder, "metadata.txt")
                with open(metadata_path, "w") as f:
                    f.write(f"Vehicle ID: {tid}\n")
                    f.write(f"Frame: {frame_count}\n")
                    f.write(f"Time (s): {frame_count / fps:.2f}\n")
                    f.write(f"Traffic Light: {traffic_state}\n")
                    f.write(f"Red Light Duration (s): {red_elapsed_time:.2f}\n")
                    f.write(f"Vehicle Box: ({x1}, {y1}, {x2}, {y2})\n")
                    f.write(f"\n--- LICENSE PLATE INFO ---\n")
                    f.write(f"Plate Detected: {plate_detected}\n")
                    if plate_detected and plate_conf > 0:
                        f.write(f"Confidence: {plate_conf:.2f}\n")
                
                state["saved"] = True
                violation_history.append({
                    "vid": tid,
                    "frame": frame_count,
                    "time": frame_count / fps,
                    "light": traffic_state,
                    "red_time": red_elapsed_time,
                    "folder": violation_folder,
                    "plate_detected": plate_detected,
                    "plate_conf": plate_conf
                })

                print(f"\n✓ Violation #{violation_counter:05d} detected!")
                print(f"  Vehicle ID: {tid} | Frame: {frame_count} | Time: {frame_count/fps:.2f}s")
                print(f"  Saved to: {violation_folder}")

            state["prev_box"] = veh_box
            
            # ═══ Store crop for sidebar display ═══
            if state["violated"]:
                vehicle_crop, crop_bbox = smart_crop_vehicle(
                    frame, 
                    x1, y1, x2, y2,
                    padding_ratio=0.2
                )
                # Sidebar hiển thị vehicle crop RAW (không CLAHE)
                current_frame_crop = vehicle_crop
            
            # ---- draw vehicle ----
            color = (0, 0, 255) if state["violated"] else (0, 255, 0)
            label = f"VIOLATION ID {tid}" if state["violated"] else f"ID {tid}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

            cv2.circle(annotated, (rear_x, rear_y), 4, (0, 0, 255), -1)

    # -------- Draw stopline --------
    for sl in current_stoplines:
        x1, y1, x2, y2 = sl
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # ========== SIDEBAR ========== 
    sidebar = np.zeros((SIDEBAR_HEIGHT, SIDEBAR_WIDTH, 3), dtype=np.uint8)
    sidebar[:] = (20, 20, 20)

    if violation_history:
        latest_vio = violation_history[-1]
        current_violation_id = latest_vio['vid']

        if last_violation_id != current_violation_id:
            last_violation_id = current_violation_id
            
            if current_frame_crop is not None:
                current_display_crop = current_frame_crop.copy()
            else:
                crop_path = os.path.join(latest_vio["folder"], "01_crop_vehicle.jpg")
                if os.path.exists(crop_path):
                    img = cv2.imread(crop_path)
                    if img is not None:
                        current_display_crop = img.copy()

        # Header
        cv2.rectangle(sidebar, (0, 0), (SIDEBAR_WIDTH, 60), (0, 0, 200), -1)
        cv2.putText(sidebar, "LATEST VIOLATION", (10, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

        # Info
        info_y = 70
        cv2.rectangle(sidebar, (0, info_y), (SIDEBAR_WIDTH, info_y + 150),
                    (40, 40, 60), -1)
        cv2.rectangle(sidebar, (0, info_y), (SIDEBAR_WIDTH, info_y + 150),
                    (0, 255, 0), 2)

        plate_status = "✓ Detected" if latest_vio['plate_detected'] else "✗ Not found"
        plate_color = (0, 255, 0) if latest_vio['plate_detected'] else (0, 0, 255)

        info_texts = [
            f"Vehicle ID: {int(latest_vio['vid'])}",
            f"Frame: {latest_vio['frame']}",
            f"Time: {latest_vio['time']:.2f}s",
            f"Light: {latest_vio['light']}",
            f"Red Duration: {latest_vio['red_time']:.2f}s",
            f"License Plate: {plate_status}"
        ]

        for idx, text in enumerate(info_texts):
            if idx == 5:
                text_color = plate_color
            else:
                text_color = (0, 0, 255) if latest_vio['light'] == "RED" else (0, 255, 255)
            
            cv2.putText(sidebar, text, (15, info_y + 25 + idx * 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)

        # Display crop
        crop_y = info_y + 190

        if current_display_crop is not None:
            try:
                crop_img = current_display_crop
                crop_h, crop_w = crop_img.shape[:2]
                max_width = SIDEBAR_WIDTH - 20
                max_height = SIDEBAR_HEIGHT - crop_y - 80

                scale = min(max_width / crop_w, max_height / crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)

                crop_resized = cv2.resize(crop_img, (new_w, new_h))

                start_x = (SIDEBAR_WIDTH - new_w) // 2
                start_y = crop_y + 10

                sidebar[start_y:start_y+new_h, start_x:start_x+new_w] = crop_resized

                cv2.rectangle(sidebar, (start_x-3, start_y-3),
                            (start_x+new_w+3, start_y+new_h+3),
                            (0, 255, 0), 2)

                cv2.putText(sidebar, "VEHICLE CROP", 
                        (start_x + 5, start_y - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1)

            except Exception as e:
                print("Error displaying crop:", e)
        
        # Statistics
        stat_y = SIDEBAR_HEIGHT - 80
        cv2.rectangle(sidebar, (0, stat_y), (SIDEBAR_WIDTH, SIDEBAR_HEIGHT),
                    (40, 40, 60), -1)
        cv2.rectangle(sidebar, (0, stat_y), (SIDEBAR_WIDTH, SIDEBAR_HEIGHT),
                    (200, 200, 0), 1)
        
        cv2.putText(sidebar, f"Total Violations: {len(violation_history)}", 
                (10, stat_y + 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        
        plates_found = sum(1 for v in violation_history if v.get("plate_detected", False))
        cv2.putText(sidebar, f"Plates Detected: {plates_found}/{len(violation_history)}", 
                (10, stat_y + 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)

    else:
        cv2.putText(sidebar, "NO VIOLATION", (20, SIDEBAR_HEIGHT // 2 - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(sidebar, "DETECTED", (20, SIDEBAR_HEIGHT // 2 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.putText(sidebar, f"Light: {traffic_state}", (20, SIDEBAR_HEIGHT - 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

    # -------- Draw info --------
    light_color = (
        (0, 0, 255) if traffic_state == "RED"
        else (0, 255, 255) if traffic_state == "YELLOW"
        else (0, 255, 0) if traffic_state == "GREEN"
        else (128, 128, 128)
    )

    cv2.putText(annotated, f"LIGHT: {traffic_state}", (40, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, light_color, 3)

    violation_count = sum(1 for v in vehicle_memory.values() if v["violated"])
    cv2.putText(annotated, f"Violations: {violation_count}", (40, 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

    # -------- RED FLASH --------
    if red_flash_remaining > 0:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        annotated = cv2.addWeighted(annotated, 1 - RED_FLASH_ALPHA,
                                    overlay, RED_FLASH_ALPHA, 0)
        red_flash_remaining -= 1

    # Write & display
    full_frame = np.hstack([annotated, sidebar])
    out.write(full_frame)
    
    preview = cv2.resize(full_frame, (1280, 720))
    try:
        cv2.imshow("Traffic Violation Detection - Plate Crop CLAHE", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass

    # -------- CLEANUP --------
    vehicles_to_remove = []
    for vid, v in vehicle_memory.items():
        if frame_count - v.get("last_seen_frame", frame_count) > MAX_IDLE_FRAMES:
            vehicles_to_remove.append(vid)
    
    for vid in vehicles_to_remove:
        del vehicle_memory[vid]

    frame_count += 1
    pbar.update(1)

pbar.close()
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("DONE!")
print(f"Total violations detected: {violation_counter}")
print(f"Output video: {OUT_VIDEO}")
print(f"Violations folder: {VIOLATIONS_DIR}")
print("=" * 50)

# ========== FINAL STATISTICS ==========
print("\n FINAL REPORT:")
plates_found = sum(1 for v in violation_history if v.get("plate_detected", False))
print(f"  Total violations: {violation_counter}")
print(f"  Plates detected: {plates_found}/{violation_counter} ({plates_found*100//violation_counter if violation_counter > 0 else 0}%)")
print(f"  Output video: {OUT_VIDEO}")
print(f"  Violations saved: {VIOLATIONS_DIR}")