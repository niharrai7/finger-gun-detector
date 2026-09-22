import cv2
import numpy as np
import time
import random
import os
import winsound
import mediapipe as mp
import config
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
HIGHSCORE_FILE = "highscore.txt"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                      # <-- two-hand support
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def is_finger_extended(lm, tip_id, pip_id):
    wrist = lm[0]
    tip = lm[tip_id]
    pip = lm[pip_id]
    return dist(wrist, tip) > dist(wrist, pip)

def get_finger_states(lm):
    index = is_finger_extended(lm, 8, 6)
    middle = is_finger_extended(lm, 12, 10)
    ring = is_finger_extended(lm, 16, 14)
    pinky = is_finger_extended(lm, 20, 18)
    return index, middle, ring, pinky

def is_finger_gun(lm):
    index, middle, ring, pinky = get_finger_states(lm)
    return index and not middle and not ring and not pinky

def is_open_palm(lm):
    index, middle, ring, pinky = get_finger_states(lm)
    return index and middle and ring and pinky

def line_hits_circle(p1, p2, center, radius):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    c = np.array(center, dtype=float)
    seg = p2 - p1
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq == 0:
        d = np.linalg.norm(c - p1)
    else:
        t = max(0, min(1, np.dot(c - p1, seg) / seg_len_sq))
        proj = p1 + t * seg
        d = np.linalg.norm(c - proj)
    return d <= radius

def load_high_score():
    if os.path.exists(HIGHSCORE_FILE):
        try:
            with open(HIGHSCORE_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0
    return 0

def save_high_score(value):
    with open(HIGHSCORE_FILE, "w") as f:
        f.write(str(value))

def play_impact_sound():
    winsound.Beep(200, 50)
    winsound.Beep(1800, 60)

def spawn_blast(pos):
    now = time.time()
    blasts.append({"pos": (int(pos[0]), int(pos[1])), "created": now})
    for _ in range(10):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(1.0, 3.5)
        smoke_particles.append({
            "pos": [float(pos[0]), float(pos[1])],
            "vel": [np.cos(angle) * speed, np.sin(angle) * speed - 0.8],
            "created": now,
            "radius": random.randint(8, 16)
        })

def make_vignette(w, h, strength=0.55):
    kx = cv2.getGaussianKernel(w, w * 0.6)
    ky = cv2.getGaussianKernel(h, h * 0.6)
    mask = ky @ kx.T
    mask = mask / mask.max()
    mask = mask * strength + (1 - strength)
    return mask.astype(np.float32)

def apply_vignette(frame, mask):
    out = frame.astype(np.float32)
    for i in range(3):
        out[:, :, i] *= mask
    return np.clip(out, 0, 255).astype(np.uint8)

def get_current_target_radius(current_score):
    # difficulty scaling: shrink targets as score climbs, floor at 15px
    return max(15, TARGET_RADIUS_BASE - current_score // 40)

def spawn_target(frame_w, frame_h, radius):
    x = random.randint(radius + 20, frame_w - radius - 20)
    y = random.randint(radius + 20, frame_h - radius - 20)
    return {"pos": (x, y), "radius": radius, "hit": False, "hit_time": 0}

# ---- Shooting/trigger state (per-hand for two-hand support) ----
prev_thumb_y = {}   # keyed by hand label ("Left"/"Right")
thumb_down_threshold = 0.03
cooldown = 0.4
last_shot_time = 0

tracers = []
muzzle_flashes = []
blasts = []
smoke_particles = []
combo_popups = []
TRACER_LIFETIME = 0.15
FLASH_LIFETIME = 0.12
BLAST_LIFETIME = 0.35
SMOKE_LIFETIME = 0.7
COMBO_POPUP_LIFETIME = 0.8
TRACER_WIDTH = 9
shot_count = 0
hit_count = 0
score = 0
hit_streak = 0

TARGET_RADIUS_BASE = 30
NUM_TARGETS = 3
targets = []

MAG_SIZE = config.MAG_SIZE
ammo = MAG_SIZE

GAME_DURATION = config.GAME_DURATION
last_reload_time = 0
RELOAD_COOLDOWN = 1.0
game_start_time = 0
high_score = load_high_score()

STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_OVER = "over"
game_state = STATE_MENU

vignette_mask = None

def reset_game(w, h):
    global shot_count, hit_count, score, hit_streak, ammo
    global tracers, muzzle_flashes, blasts, smoke_particles, combo_popups
    global targets, game_start_time, prev_thumb_y
    shot_count = 0
    hit_count = 0
    score = 0
    hit_streak = 0
    ammo = MAG_SIZE
    tracers = []
    muzzle_flashes = []
    blasts = []
    smoke_particles = []
    combo_popups = []
    targets = [spawn_target(w, h, TARGET_RADIUS_BASE) for _ in range(NUM_TARGETS)]
    game_start_time = time.time()
    prev_thumb_y = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if vignette_mask is None:
        vignette_mask = make_vignette(w, h)

    now = time.time()

    # ---------------- MENU STATE ----------------
    if game_state == STATE_MENU:
        frame = apply_vignette(frame, vignette_mask)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, "FINGER GUN DETECTOR", (w // 2 - 280, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        cv2.putText(frame, "Press SPACE to start", (w // 2 - 200, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"High Score: {high_score}", (w // 2 - 130, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Finger Gun Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            reset_game(w, h)
            game_state = STATE_PLAYING
        continue

    time_left = max(0, GAME_DURATION - (now - game_start_time))
    if time_left <= 0 and game_state == STATE_PLAYING:
        if score > high_score:
            high_score = score
            save_high_score(high_score)
        game_state = STATE_OVER

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if game_state == STATE_PLAYING and result.hand_landmarks:
        for idx, lm in enumerate(result.hand_landmarks):
            if result.handedness and len(result.handedness) > idx:
                hand_label = result.handedness[idx][0].category_name
            else:
                hand_label = f"hand{idx}"

            for point in lm:
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            gun_pose = is_finger_gun(lm)
            palm_open = is_open_palm(lm)

            if palm_open and ammo < MAG_SIZE and (now - last_reload_time) > RELOAD_COOLDOWN:
                ammo = MAG_SIZE
                last_reload_time = now
                winsound.Beep(600, 150)

            if gun_pose:
                thumb_tip_y = lm[4].y
                prev_y = prev_thumb_y.get(hand_label)
                if prev_y is not None:
                    thumb_drop = thumb_tip_y - prev_y
                    if thumb_drop > thumb_down_threshold and (now - last_shot_time) > cooldown:
                        if ammo > 0:
                            last_shot_time = now
                            shot_count += 1
                            ammo -= 1
                            winsound.Beep(1000, 80)

                            wrist = lm[0]
                            tip = lm[8]
                            wx, wy = wrist.x * w, wrist.y * h
                            tx, ty = tip.x * w, tip.y * h
                            dx, dy = tx - wx, ty - wy
                            length = max((dx**2 + dy**2) ** 0.5, 1)
                            dx, dy = dx / length, dy / length
                            end_x = int(tx + dx * 1000)
                            end_y = int(ty + dy * 1000)

                            tracers.append({
                                "start": (int(tx), int(ty)),
                                "end": (end_x, end_y),
                                "created": now
                            })
                            muzzle_flashes.append({
                                "pos": (int(tx), int(ty)),
                                "created": now
                            })

                            shot_hit = False
                            for t in targets:
                                if not t["hit"] and line_hits_circle(
                                    (tx, ty), (end_x, end_y), t["pos"], t["radius"]
                                ):
                                    t["hit"] = True
                                    t["hit_time"] = now
                                    score += 10
                                    shot_hit = True
                                    spawn_blast(t["pos"])
                                    play_impact_sound()

                            if shot_hit:
                                hit_count += 1
                                hit_streak += 1
                                if hit_streak % 3 == 0:
                                    score += 15
                                    combo_popups.append({
                                        "text": f"COMBO x{hit_streak}! +15",
                                        "created": now
                                    })
                            else:
                                hit_streak = 0
                        else:
                            winsound.Beep(300, 100)

                prev_thumb_y[hand_label] = thumb_tip_y
            else:
                prev_thumb_y[hand_label] = None
    elif game_state == STATE_PLAYING:
        cv2.putText(frame, "SHOW YOUR HAND", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if game_state == STATE_PLAYING:
        # respawn/draw targets (with difficulty-scaled radius)
        for t in targets:
            if t["hit"]:
                if now - t["hit_time"] > 0.3:
                    new_radius = get_current_target_radius(score)
                    new_t = spawn_target(w, h, new_radius)
                    t["pos"] = new_t["pos"]
                    t["radius"] = new_t["radius"]
                    t["hit"] = False
                else:
                    cv2.circle(frame, t["pos"], t["radius"], (0, 255, 0), -1)
            else:
                cv2.circle(frame, t["pos"], t["radius"], (0, 0, 255), 3)
                cv2.circle(frame, t["pos"], 6, (0, 0, 255), -1)

        # tracers
        active_tracers = []
        for t in tracers:
            age = now - t["created"]
            if age < TRACER_LIFETIME:
                alpha = 1 - (age / TRACER_LIFETIME)
                thickness = max(2, int(TRACER_WIDTH * alpha))
                cv2.line(frame, t["start"], t["end"], (0, 255, 255), thickness)
                active_tracers.append(t)
        tracers = active_tracers

        # muzzle flashes
        active_flashes = []
        for f in muzzle_flashes:
            age = now - f["created"]
            if age < FLASH_LIFETIME:
                progress = age / FLASH_LIFETIME
                radius = int(10 + progress * 25)
                alpha = 1 - progress
                color = (0, int(165 * alpha + 90), int(255 * alpha))
                cv2.circle(frame, f["pos"], radius, color, 2)
                cv2.circle(frame, f["pos"], max(2, int(6 * (1 - progress))), (0, 255, 255), -1)
                active_flashes.append(f)
        muzzle_flashes = active_flashes

        # blast rings
        active_blasts = []
        for b in blasts:
            age = now - b["created"]
            if age < BLAST_LIFETIME:
                progress = min(max(age / BLAST_LIFETIME, 0.0), 1.0)
                radius = max(1, int(15 + progress * 55))
                thickness = max(1, min(20, int(6 * (1 - progress))))
                g_val = max(0, min(255, int(140 + 100 * (1 - progress))))
                color = (0, g_val, 255)
                center = (int(b["pos"][0]), int(b["pos"][1]))
                cv2.circle(frame, center, radius, color, thickness)
                active_blasts.append(b)
        blasts = active_blasts

        # smoke particles
        active_smoke = []
        if smoke_particles:
            smoke_overlay = frame.copy()
            for p in smoke_particles:
                age = now - p["created"]
                if age < SMOKE_LIFETIME:
                    p["pos"][0] += p["vel"][0]
                    p["pos"][1] += p["vel"][1]
                    p["vel"][1] -= 0.03
                    progress = min(max(age / SMOKE_LIFETIME, 0.0), 1.0)
                    radius = max(1, int(p["radius"] + progress * 10))
                    gray = max(0, min(255, int(160 - progress * 60)))
                    center = (int(p["pos"][0]), int(p["pos"][1]))
                    cv2.circle(smoke_overlay, center, radius, (gray, gray, gray), -1)
                    active_smoke.append(p)
            if active_smoke:
                frame = cv2.addWeighted(smoke_overlay, 0.35, frame, 0.65, 0)
        smoke_particles = active_smoke

        # combo popups
        active_popups = []
        for cpop in combo_popups:
            age = now - cpop["created"]
            if age < COMBO_POPUP_LIFETIME:
                alpha = 1 - (age / COMBO_POPUP_LIFETIME)
                y_offset = int(age * 40)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.putText(frame, cpop["text"], (w // 2 - 150, 100 - y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                active_popups.append(cpop)
        combo_popups = active_popups

        accuracy = (hit_count / shot_count * 100) if shot_count > 0 else 0.0

        cv2.putText(frame, f"Shots: {shot_count}", (50, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score}", (50, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (50, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Time: {int(time_left)}s", (w - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Ammo: {ammo}/{MAG_SIZE}", (w - 220, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        frame = apply_vignette(frame, vignette_mask)

    if game_state == STATE_OVER:
        accuracy = (hit_count / shot_count * 100) if shot_count > 0 else 0.0
        frame = apply_vignette(frame, vignette_mask)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "GAME OVER", (w // 2 - 150, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"Score: {score}", (w // 2 - 100, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (w // 2 - 130, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"High Score: {high_score}", (w // 2 - 130, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press R to restart or Q to quit", (w // 2 - 220, h // 2 + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Finger Gun Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r') and game_state == STATE_OVER:
        game_state = STATE_MENU

cap.release()
cv2.destroyAllWindows()