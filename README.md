# TRAFFIC-ANALYSIS
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ── Models ──────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# ── Video ───────────────────────────────────────────────
cap = cv2.VideoCapture("traffic.mp4")
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video    = int(cap.get(cv2.CAP_PROP_FPS))
lane_width   = frame_width // 3

# ── Video Writer (saves output) ──────────────────────────
out = cv2.VideoWriter("output.avi",
      cv2.VideoWriter_fourcc(*"XVID"),
      fps_video, (frame_width, frame_height))

# ── Constants ────────────────────────────────────────────
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}
SWITCH_EVERY     = 30
EMERGENCY_DURATION = 90
RED_PIXEL_THRESHOLD = 5000

# ── State ────────────────────────────────────────────────
current_green   = "Lane 1"
frame_counter   = 0
emergency_active = False
emergency_timer  = 0
prev_time        = time.time()

lane_wait_time  = {"Lane 1": 0, "Lane 2": 0, "Lane 3": 0}
total_vehicles_seen = 0

# ── Functions ────────────────────────────────────────────

def decide_green_lane(lane_counts, bus_lanes):
    """Bus gets priority. Otherwise busiest lane gets green."""
    if bus_lanes:
        return bus_lanes[0]
    return max(lane_counts, key=lane_counts.get)

def detect_ambulance_by_color(frame):
    """Detect large red vehicle = ambulance simulation."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   100, 100), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    return cv2.countNonZero(mask1 + mask2) > RED_PIXEL_THRESHOLD

def draw_signals(frame, green_lane, emergency):
    """Draw RED/GREEN signal boxes at bottom of each lane."""
    lanes = ["Lane 1", "Lane 2", "Lane 3"]
    for i, lane in enumerate(lanes):
        if emergency:
            color = (0, 255, 0)
            label = "EMERGENCY"
        else:
            color = (0, 255, 0) if lane == green_lane else (0, 0, 255)
            label = "GREEN"     if lane == green_lane else "RED"
        x = i * lane_width + 10
        cv2.rectangle(frame, (x, frame_height - 60),
                      (x + 150, frame_height - 10), color, -1)
        cv2.putText(frame, f"{lane}: {label}",
                    (x + 5, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)

def draw_dashboard(frame, total, lane_counts, type_counts,
                   lane_wait_time, current_green, fps,
                   emergency_active, bus_priority):
    """Draw all stats on left side of frame."""
    # Background panel
    cv2.rectangle(frame, (0, 0), (320, 260), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (320, 260), (255,255,255), 1)

    y = 25
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    y += 25
    # Total
    cv2.putText(frame, f"Total Vehicles: {total}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    y += 25
    # Type breakdown
    cv2.putText(frame, f"Cars:{type_counts['car']}  Bus:{type_counts['bus']}  Trucks:{type_counts['truck']}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,200,0), 2)
    y += 25
    # Lane counts
    cv2.putText(frame, f"L1:{lane_counts['Lane 1']}  L2:{lane_counts['Lane 2']}  L3:{lane_counts['Lane 3']}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    y += 25
    # Wait times
    cv2.putText(frame, f"Wait(f) L1:{lane_wait_time['Lane 1']}  L2:{lane_wait_time['Lane 2']}  L3:{lane_wait_time['Lane 3']}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)
    y += 25
    # Green lane
    if not emergency_active:
        label = f"GREEN: {current_green}"
        if bus_priority:
            label += " (BUS PRIORITY)"
        cv2.putText(frame, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    y += 25
    # Pedestrian warning handled separately

def draw_emergency_banner(frame, frame_width):
    """Red banner across top when emergency active."""
    cv2.rectangle(frame, (0, 265), (frame_width, 315), (0,0,255), -1)
    cv2.putText(frame, "EMERGENCY VEHICLE DETECTED - GREEN WAVE ACTIVE",
                (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255,255,255), 2)

def draw_pedestrian_warning(frame, frame_width):
    """Yellow banner when pedestrian detected."""
    cv2.rectangle(frame, (0, 265), (frame_width, 315), (0,165,255), -1)
    cv2.putText(frame, "PEDESTRIAN DETECTED - EXTENDING CROSSING TIME",
                (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255,255,255), 2)

# ── Main Loop ────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # Run YOLO
    results = model(frame, verbose=False)

    # Reset counts
    lane_counts  = {"Lane 1": 0, "Lane 2": 0, "Lane 3": 0}
    type_counts  = {"car": 0, "bus": 0, "truck": 0}
    bus_lanes    = []
    pedestrian_detected = False
    frame_vehicles = 0

    for box in results[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2

        # Pedestrian detection
        if cls == 0:
            pedestrian_detected = True
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,165,255), 2)
            cv2.putText(frame, f"person {conf:.2f}",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)
            continue

        # Vehicle detection
        if cls in VEHICLE_CLASSES:
            frame_vehicles += 1
            vtype = VEHICLE_CLASSES[cls]
            type_counts[vtype] += 1

            # Lane assignment
            if cx < lane_width:
                lane = "Lane 1"
            elif cx < lane_width * 2:
                lane = "Lane 2"
            else:
                lane = "Lane 3"
            lane_counts[lane] += 1

            # Track bus lanes for priority
            if cls == 5:
                bus_lanes.append(lane)

            # Draw box
            color = (0,255,0)
            if cls == 5:
                color = (255,165,0)  # orange for bus
            elif cls == 7:
                color = (0,255,255)  # yellow for truck
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"{vtype} {conf:.2f}",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    total_vehicles_seen += frame_vehicles

    # Emergency detection
    if detect_ambulance_by_color(frame):
        emergency_active = True
        emergency_timer  = EMERGENCY_DURATION

    if emergency_timer > 0:
        emergency_timer  -= 1
        emergency_active  = True
    else:
        emergency_active  = False

    # Signal decision every 30 frames
    frame_counter += 1
    bus_priority = len(bus_lanes) > 0
    if frame_counter % SWITCH_EVERY == 0 and not emergency_active:
        current_green = decide_green_lane(lane_counts, bus_lanes)

    # Wait time tracking
    for lane in lane_wait_time:
        if lane != current_green and not emergency_active:
            lane_wait_time[lane] += 1
        else:
            lane_wait_time[lane] = 0

    # Draw lane dividers
    cv2.line(frame, (lane_width,   0), (lane_width,   frame_height), (255,0,0), 2)
    cv2.line(frame, (lane_width*2, 0), (lane_width*2, frame_height), (255,0,0), 2)

    # Draw everything
    draw_signals(frame, current_green, emergency_active)
    draw_dashboard(frame, frame_vehicles, lane_counts, type_counts,
                   lane_wait_time, current_green, fps,
                   emergency_active, bus_priority)

    if emergency_active:
        draw_emergency_banner(frame, frame_width)
    elif pedestrian_detected:
        draw_pedestrian_warning(frame, frame_width)

    # Save frame to output video
    out.write(frame)

    cv2.imshow("AI Traffic Control System", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Total vehicles detected in session: {total_vehicles_seen}")
print("Output saved to output.avi")
