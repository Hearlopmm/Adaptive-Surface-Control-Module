#!/usr/bin/env python3
import time
import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
from ultralytics import YOLO
import cv2
from opencv_init import initial as init
import json
import paho.mqtt.client as mqtt
# from paho.mqtt.client import CallbackAPIVersion


# =========================== voice part ===============================
# ====== CONFIG ======
MIC = 3
FS = 44100
FS_VAD = 16000
FRAME_MS = 30
HOP = int(FS * FRAME_MS / 1000)
HOP_VAD = int(FS_VAD * FRAME_MS / 1000)
REF_RMS = 5e-4
REF_DB = 40.0
VAD_MODE = 2
DB_CUT = 45
WIN_SEC = 5
RATIO_CUT = 0.3
PRINT_DT = 0.5
ON_TIME = 3
OFF_TIME = 3
# ====== STATE ======
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)
buf = deque()
t_all = 0.0
t_talk = 0.0
t_last = 0.0
speech_flag = False
t_on = 0.0
t_off = 0.0


def rms2db(r):
    if r <= 0:
        return -120
    return REF_DB + 20 * np.log10(r / REF_RMS)


def simple_resample(x, n_out):
    if len(x) == 0:
        return np.zeros(n_out, np.float32)
    return np.interp(
        np.linspace(0, 1, n_out, endpoint=False),
        np.linspace(0, 1, len(x), endpoint=False),
        x
    ).astype(np.float32)


def audio_cb(indata, frames, info, status):
    global buf, t_all, t_talk, t_last
    global speech_flag, t_on, t_off

    now = time.time()
    dt = frames / float(FS)

    x = indata[:, 0].astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    db = rms2db(rms)

    x_vad = simple_resample(x, HOP_VAD)
    x_i16 = (x_vad * 32768).astype(np.int16)
    vad_raw = vad.is_speech(x_i16.tobytes(), FS_VAD)

    is_talk = bool(vad_raw and db >= DB_CUT)

    buf.append((now, dt, is_talk))
    t_all += dt
    if is_talk:
        t_talk += dt

    # clean window
    cutoff = now - WIN_SEC
    while buf and buf[0][0] < cutoff:
        _, dt_old, tk_old = buf.popleft()
        t_all -= dt_old
        if tk_old:
            t_talk -= dt_old

    ratio = t_talk / t_all if t_all > 0 else 0
    result = 1 if ratio >= RATIO_CUT else 0

    # long-term flag
    if result == 1:
        t_on += dt
        t_off = 0
        if not speech_flag and t_on >= ON_TIME:
            speech_flag = True
    else:
        t_on = 0
        t_off += dt
        if speech_flag and t_off >= OFF_TIME:
            speech_flag = False

    # # periodic print
    # if now - t_last >= PRINT_DT:
    #     print(speech_flag)
    #     t_last = now


# ============================= cv part =====================================
# ---DEFINE---
model = YOLO("./yolov8n.pt")
GREEN = (0, 255, 0)
PINK = (255, 105, 180)
person_conf_threshold = 0.45
INFER_INTERVAL = 4  # second
H_FULL_PATH = "H_full.npy"
H_full = np.load(H_FULL_PATH)
scales = [
    (808, 1225, 946, 1381),  # box 1
    (720, 1381, 850, 1458),  # box 2
    (871, 1380, 1010, 1458), # box 3
]

# ---Initialization---
# cap = cv2.VideoCapture(0)
cap, img_w, img_h = init.cap_initial(0)
print(img_w, img_h)
pTime = cTime = 0
# frame_id = 0
last_infer_time = 0.0
last_results = None
last_boxes = None
showfig = True
move_motors = []

def is_merge(a, b, near_thresh=30, area_low=0.65, area_high=1.35):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    ratio = area_a / area_b
    if not (area_low <= ratio <= area_high):
        return False
    # --- Rule 1: overlap ---
    overlap = not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)
    if overlap:
        return True
    # --- Rule 2: near horizontally ---
    # A left B right
    if 0 <= bx1 - ax2 < near_thresh:
        return True
    # B left A right
    if 0 <= ax1 - bx2 < near_thresh:
        return True
    return False

def get_fps(show=False, inimg=None):
    global pTime, cTime
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    if show and inimg is not None:
        cv2.putText(inimg, f"fps: {int(fps)}", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
    return fps

def get_motors(positions, H_matrix, motor_boxes):
    targets = []
    for fx, fy in positions:
        tx, ty = init.map_point(fx, fy, H_matrix)
        motor_id = init.which_box(tx, ty, motor_boxes)
        if motor_id is not None and motor_id not in targets:
            targets.append(motor_id)
    return targets

# ============================= COMM part =================================
send_info = None

# open==ture, close=false
motor1status = False
motor2status = False
motor3status = False

MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC       = "demo/number"
MQTT_CLIENT_ID   = "pi4_keyboard_sender"

client = mqtt.Client(client_id=MQTT_CLIENT_ID)
client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=30)
client.loop_start()
print("Pi 4B sender started.")

# =============================== RUN ================================
with sd.InputStream(
    device=MIC,
    channels=1,
    samplerate=FS,
    blocksize=HOP,
    dtype="int16",
    callback=audio_cb,
):
    try:
        last_speech_flag = True
        while True:
            if speech_flag != last_speech_flag:
                print("Speeching flag change to:", speech_flag)
                last_speech_flag = speech_flag

            # people detect
            ret, frame = cap.read()
            # frame = init.imgResize(frame, 320)
            if not ret:
                break
            now = loop_start = time.time()
            if (last_results is None) or (now - last_infer_time > INFER_INTERVAL):
                results = model(frame, verbose=False)[0]
                boxes = results.boxes
                last_results = results
                last_boxes = boxes
                last_infer_time = now
            else:
                results = last_results
                boxes = last_boxes

            person_boxes = []
            single_positions = []  # 保存所有单人的脚中点坐标

            # Step 1:  person + conf>=conf_threshold
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != 0 or conf < person_conf_threshold:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                person_boxes.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf
                })

            # S2: combine in a group for multiple people
            groups = []
            visited = set()
            for i in range(len(person_boxes)):
                if i in visited:
                    continue
                group = [i]
                visited.add(i)
                for j in range(len(person_boxes)):
                    if j in visited:
                        continue
                    if is_merge(person_boxes[i]["bbox"], person_boxes[j]["bbox"],
                                near_thresh=60, area_low=0.5, area_high=1.7):
                        group.append(j)
                        visited.add(j)
                groups.append(group)

            for group in groups:

                if len(group) == 1:
                    # ------- 单独人：绿色 -------
                    idx = group[0]
                    x1, y1, x2, y2 = map(int, person_boxes[idx]["bbox"])
                    conf = person_boxes[idx]["conf"]

                    # 画人的框
                    if showfig:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                        cv2.putText(frame, f"person {conf:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

                    cx = (x1 + x2) // 2
                    cy = y2

                    if showfig:
                        cv2.circle(frame, (cx, cy), 4, GREEN, -1)
                        coord_text = f"({cx}, {cy})"
                        cv2.putText(frame, coord_text,
                                    (cx - 60, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

                    single_positions.append((cx, cy))

                else:
                    xs1, ys1, xs2, ys2 = [], [], [], []
                    for idx in group:
                        x1, y1, x2, y2 = person_boxes[idx]["bbox"]
                        xs1.append(x1)
                        ys1.append(y1)
                        xs2.append(x2)
                        ys2.append(y2)

                    gx1, gy1 = int(min(xs1)), int(min(ys1))
                    gx2, gy2 = int(max(xs2)), int(max(ys2))

                    if showfig:
                        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), PINK, 3)
                        cv2.putText(frame,
                                    f"{len(group)} persons",
                                    (gx1, gy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, PINK, 2)

                    for idx in group:
                        x1, y1, x2, y2 = map(int, person_boxes[idx]["bbox"])
                        cx = (x1 + x2) // 2
                        cy = y2

                        if showfig:
                            cv2.circle(frame, (cx, cy), 4, PINK, -1)
                            coord_text = f"({cx}, {cy})"
                            cv2.putText(frame, coord_text,
                                        (cx - 60, cy - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PINK, 1)

            fps = init.get_fps(showfig, frame)

            if showfig:
                cv2.imshow("People Overlap Merge", frame)
                if cv2.waitKey(1) == 27:
                    break

            # 计算电机目标
            move_motors = get_motors(single_positions, H_full, scales)
            # if move_motors != [0]:
                # print("move_motors:", move_motors)

            # =================== send_info ===================
            send_info = None
            if speech_flag:
                if motor1status or motor2status or motor3status:
                    send_info = 0
                    motor1status = False
                    motor2status = False
                    motor3status = False
            else:
                desired_on = set(move_motors)
                changed = []
                # motor 1
                desired1 = 1 in desired_on
                if motor1status != desired1:
                    changed.append(1)
                    motor1status = desired1
                # motor 2
                desired2 = 2 in desired_on
                if motor2status != desired2:
                    changed.append(2)
                    motor2status = desired2
                # motor 3
                desired3 = 3 in desired_on
                if motor3status != desired3:
                    changed.append(3)
                    motor3status = desired3
                if changed:
                    # 1,2,3,12,13,23,123
                    send_info = int("".join(str(i) for i in sorted(changed)))

            if send_info is not None:
                payload = str(send_info)
                client.publish(MQTT_TOPIC, payload, qos=0, retain=False)
                print("send_info:", send_info)

    except KeyboardInterrupt:
        pass

cap.release()
cv2.destroyAllWindows()
client.loop_stop()
client.disconnect()
print("MQTT COMM ended.")
