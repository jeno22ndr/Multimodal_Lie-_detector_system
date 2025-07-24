import cv2
import mediapipe as mp
import numpy as np
import time
import math
import threading
import queue
from scipy.spatial import distance as dist
from fer import FER
from scipy.signal import find_peaks, butter, filtfilt
from collections import deque
from deepface import DeepFace
import librosa
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip
import sounddevice as sd

# =================================================================================
# --------------------------- Configuration & Globals ---------------------------
# =================================================================================

# --- Analysis Configuration ---
CALIBRATION_SECONDS = 15  # Duration to establish a baseline
MAX_FRAMES_BPM = 300      # Buffer size for heart rate calculation (10 seconds at 30fps)
EAR_THRESHOLD = 0.21      # Eye Aspect Ratio threshold for blink detection
MAR_COMPRESSION_THRESHOLD = 0.25 # Mouth Aspect Ratio for lip compression
MAR_VARIANCE_THRESHOLD = 0.005   # Mouth Aspect Ratio variance for speech detection

# --- Threading and State Management ---
analysis_results = {
    "Face Status": "No Face Detected",
    "Deception Likelihood": 0,
    "Dominant Emotion": "N/A",
    "Speech Emotion": "N/A",
    "Heart Rate (BPM)": "Calibrating...",
    "Respiration Rate": "Calibrating...",
    "Blink Rate": "Calibrating...",
    "Hand Position": "N/A",
    "Head Movement": "N/A",
    "Eye Gaze": "N/A",
    "Lip Compression": "N/A",
    "Microexpressions": "N/A",
    "Vocal Pitch": "N/A"
}
analysis_queue = queue.Queue() # Queue for frames to be processed by heavy models
results_lock = threading.Lock()
paused = False
playback_speed = 1

# --- Baselines (Established during calibration) ---
baselines = {
    "bpm": [], "respiration": [], "blink_rate": [], "pitch": []
}
calibrated_baselines = {
    "bpm": 75, "respiration": 16, "blink_rate": 20, "pitch": 150
}
calibration_complete = False

# --- Data Buffers ---
rppg_buffer = deque(maxlen=MAX_FRAMES_BPM)
mar_window = deque(maxlen=15)
blink_timestamps = deque()

# =================================================================================
# --------------------------- Model & Asset Loading -----------------------------
# =================================================================================

# --- Load Models ---
print("Loading models, this may take a moment...")
try:
    # refine_landmarks=True is crucial for iris tracking
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=2, min_detection_confidence=0.7
    )
    emotion_detector = FER(mtcnn=True)
    # The sound model path needs to be valid.
    # sound_model = load_model("path/to/your/sound_model.h5")
    # sound_emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
except Exception as e:
    print(f"Error loading a model: {e}")
    print("Please ensure all model files are correctly located.")
    mp_face_mesh = mp_hands = emotion_detector = None

# --- Load Meter Image ---
try:
    meter_image = cv2.imread('meter.png') # Ensure meter.png is in the same directory
    if meter_image is None:
        raise FileNotFoundError("meter.png not found")
except Exception as e:
    print(f"Could not load meter.png: {e}. The meter will not be displayed.")
    meter_image = None

# =================================================================================
# --------------------------- Core Analysis Functions ---------------------------
# =================================================================================

def euclidean(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def compute_EAR(landmarks, indices, image_width, image_height):
    points = [(int(landmarks[idx].x * image_width), int(landmarks[idx].y * image_height)) for idx in indices]
    vertical1 = euclidean(points[1], points[5])
    vertical2 = euclidean(points[2], points[4])
    horizontal = euclidean(points[0], points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0

def compute_MAR(landmarks, image_width, image_height):
    ul = landmarks[13]
    ll = landmarks[14]
    lc = landmarks[61]
    rc = landmarks[291]
    vertical_dist = euclidean((ul.x, ul.y), (ll.x, ll.y))
    horizontal_dist = euclidean((lc.x, rc.x), (rc.y, rc.y))
    return vertical_dist / horizontal_dist if horizontal_dist != 0 else 0

def analyze_eye_gaze(face_landmarks):
    """Analyzes the direction of eye gaze using iris landmarks."""
    try:
        # Right eye (person's right, screen's left)
        r_eye_h_left = face_landmarks[362]
        r_eye_h_right = face_landmarks[263]
        r_iris = face_landmarks[473]
        
        # Left eye (person's left, screen's right)
        l_eye_h_left = face_landmarks[133]
        l_eye_h_right = face_landmarks[33]
        l_iris = face_landmarks[468]

        r_eye_h_range = r_eye_h_right.x - r_eye_h_left.x
        r_h_ratio = (r_iris.x - r_eye_h_left.x) / r_eye_h_range if r_eye_h_range != 0 else 0.5
        
        l_eye_h_range = l_eye_h_right.x - l_eye_h_left.x
        l_h_ratio = (l_iris.x - l_eye_h_left.x) / l_eye_h_range if l_eye_h_range != 0 else 0.5

        avg_h_ratio = (r_h_ratio + l_h_ratio) / 2.0

        if avg_h_ratio > 0.6:
            return "Looking Right"
        elif avg_h_ratio < 0.4:
            return "Looking Left"
        else:
            return "Forward"
    except Exception:
        return "N/A"

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def calculate_rppg_vitals(buffer, fs):
    if len(buffer) < MAX_FRAMES_BPM:
        return None, None
    signal = np.array(buffer)
    hr_filtered = butter_bandpass_filter(signal, 0.75, 4.0, fs)
    hr_peaks, _ = find_peaks(hr_filtered, height=np.std(hr_filtered) * 0.5, distance=fs*0.4)
    bpm = int(np.median(60.0 / (np.diff(hr_peaks) / fs))) if len(hr_peaks) >= 2 else None
    resp_filtered = butter_bandpass_filter(signal, 0.1, 0.5, fs)
    resp_peaks, _ = find_peaks(resp_filtered, height=np.std(resp_filtered) * 0.5, distance=fs*1.5)
    respiration = int(np.median(60.0 / (np.diff(resp_peaks) / fs))) if len(resp_peaks) >= 2 else None
    return bpm, respiration

def calculate_deception_likelihood(results, baseline):
    if results["Face Status"] != "Face Detected":
        return 0
    base_score = 0
    weights = {
        "emotion_mismatch": 30, "stress_emotion": 15, "bpm_increase": 25,
        "blink_rate_change": 10, "hand_on_face": 25, "lip_compression": 15,
        "head_movement": 10, "pitch_increase": 20, "gaze_aversion": 15
    }
    face_emo = results.get("Dominant Emotion", "neutral").lower()
    speech_emo = results.get("Speech Emotion", "neutral").lower()
    stress_emotions = ["fear", "angry", "disgust", "sad"]
    if face_emo in stress_emotions and speech_emo not in stress_emotions and speech_emo != "n/a":
        base_score += weights["emotion_mismatch"]
    elif face_emo in stress_emotions:
        base_score += weights["stress_emotion"]
    try:
        current_bpm = int(results.get("Heart Rate (BPM)", baseline["bpm"]))
        if current_bpm > baseline["bpm"] * 1.15:
            base_score += weights["bpm_increase"] * min((current_bpm / baseline["bpm"] - 1.15), 1.0)
    except (ValueError, TypeError): pass
    try:
        current_blink_rate = int(results.get("Blink Rate", baseline["blink_rate"]))
        if current_blink_rate > baseline["blink_rate"] * 1.5 or current_blink_rate < baseline["blink_rate"] * 0.5:
             base_score += weights["blink_rate_change"]
    except (ValueError, TypeError): pass
    try:
        current_pitch = float(results.get("Vocal Pitch", baseline["pitch"]).replace(" Hz", ""))
        if current_pitch > baseline["pitch"] * 1.20:
            base_score += weights["pitch_increase"]
    except (ValueError, TypeError): pass
    if results.get("Eye Gaze", "Forward") != "Forward":
        base_score += weights["gaze_aversion"]
    if results.get("Hand Position") == "Hand on Face/Mouth":
        base_score += weights["hand_on_face"]
    if results.get("Lip Compression") == "Detected":
        base_score += weights["lip_compression"]
    if "Rapid" in results.get("Head Movement", ""):
        base_score += weights["head_movement"]
    likelihood = min(base_score, 95)
    return int(likelihood)

# =================================================================================
# --------------------------- Background Analysis Thread ------------------------
# =================================================================================

def heavy_analysis_worker(video_clip):
    global analysis_results, results_lock, calibration_complete, calibrated_baselines, baselines
    last_nose = None
    last_speech_analysis_time = 0
    while True:
        try:
            frame_rgb, face_landmarks, hands_landmarks, current_time_sec = analysis_queue.get(timeout=1)
        except queue.Empty:
            continue
        local_results = {}
        try:
            detected_mood, score = emotion_detector.top_emotion(frame_rgb)
            local_results["Dominant Emotion"] = detected_mood if score and score > 0.4 else "Neutral"
        except Exception:
            local_results["Dominant Emotion"] = "N/A"
        if current_time_sec - last_speech_analysis_time > 2.0:
            last_speech_analysis_time = current_time_sec
            try:
                audio_segment = video_clip.audio.subclip(current_time_sec, current_time_sec + 2.0).to_soundarray(fps=22050)
                audio_segment_mono = librosa.to_mono(audio_segment.T)
                f0, _, _ = librosa.pyin(audio_segment_mono, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                valid_f0 = [f for f in f0 if f > 0]
                pitch = np.mean(valid_f0) if valid_f0 else calibrated_baselines.get("pitch", 0)
                local_results["Vocal Pitch"] = f"{pitch:.2f} Hz"
                if not calibration_complete: baselines["pitch"].append(pitch)
                local_results["Speech Emotion"] = "neutral"
            except Exception:
                local_results["Speech Emotion"] = "N/A"
                local_results["Vocal Pitch"] = "N/A"
        hand_on_face = False
        if hands_landmarks and face_landmarks:
            for hand in hands_landmarks:
                for i in [0, 4, 8, 12, 16, 20]:
                    hand_x, hand_y = hand.landmark[i].x, hand.landmark[i].y
                    for j in [1, 13, 14, 50, 280]:
                        face_x, face_y = face_landmarks[j].x, face_landmarks[j].y
                        if euclidean((hand_x, hand_y), (face_x, face_y)) < 0.1:
                            hand_on_face = True
                            break
                    if hand_on_face: break
        local_results["Hand Position"] = "Hand on Face/Mouth" if hand_on_face else "No Hand Covering"
        current_nose = (face_landmarks[1].x, face_landmarks[1].y)
        if last_nose:
            delta = euclidean(current_nose, last_nose)
            local_results["Head Movement"] = "Rapid" if delta > 0.015 else "Stable"
        last_nose = current_nose
        with results_lock:
            analysis_results.update(local_results)
        analysis_queue.task_done()

# =================================================================================
# -------------------------------- UI & Drawing ---------------------------------
# =================================================================================

def draw_overlay(frame, results, is_calibrating):
    h, w, _ = frame.shape
    overlay = frame.copy()

    # --- Deception Meter at Top Center ---
    if meter_image is not None and not is_calibrating:
        meter_w, meter_h = 300, 25
        meter_x_start = (w - meter_w) // 2
        meter_y_start = 20
        try:
            resized_meter = cv2.resize(meter_image, (meter_w, meter_h))
            overlay[meter_y_start : meter_y_start + meter_h, meter_x_start : meter_x_start + meter_w] = resized_meter
            likelihood = results.get("Deception Likelihood", 0)
            pointer_pos_x = meter_x_start + int((likelihood / 100.0) * (meter_w - 4))
            pts = np.array([[pointer_pos_x, meter_y_start + meter_h], [pointer_pos_x - 6, meter_y_start + meter_h + 10], [pointer_pos_x + 6, meter_y_start + meter_h + 10]], np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.putText(overlay, f"{likelihood}%", (meter_x_start + meter_w + 10, meter_y_start + meter_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception: pass

    # --- Info Panel at Bottom Left ---
    panel_h, panel_w = 240, 280
    panel_x, panel_y = 10, h - panel_h - 10
    
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if is_calibrating:
        cv2.putText(frame, "CALIBRATING...", (panel_x + 10, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "Please remain neutral.", (panel_x + 10, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    cv2.putText(frame, "REAL-TIME ANALYSIS", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 255), 1)
    cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_w - 10, panel_y + 35), (0, 255, 255), 1)
    
    y_pos = panel_y + 60
    def draw_text(label, value, y):
        cv2.putText(frame, f"{label}: {value}", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return y + 22
    
    y_pos = draw_text("Face Emotion", results.get("Dominant Emotion"), y_pos)
    y_pos = draw_text("Vocal Pitch", results.get("Vocal Pitch"), y_pos)
    y_pos = draw_text("Heart Rate", f"{results.get('Heart Rate (BPM)')} (B: {calibrated_baselines['bpm']})", y_pos)
    y_pos = draw_text("Blink Rate", f"{results.get('Blink Rate')}/min (B: {calibrated_baselines['blink_rate']})", y_pos)
    y_pos = draw_text("Eye Gaze", results.get("Eye Gaze"), y_pos)
    y_pos = draw_text("Hand Position", results.get("Hand Position"), y_pos)
    y_pos = draw_text("Lip Compress", results.get("Lip Compression"), y_pos)
    y_pos = draw_text("Head Movement", results.get("Head Movement"), y_pos)
    return frame

# =================================================================================
# -------------------------------- Main Function --------------------------------
# =================================================================================

def main():
    global paused, playback_speed, calibration_complete, calibrated_baselines, analysis_results, blink_timestamps
    video_path = r"C:\Users\jeno22ndr\OneDrive\Documents\Desktop\Temp\Boeing.mp4"
    try:
        cap = cv2.VideoCapture(video_path)
        video_clip = VideoFileClip(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened(): raise IOError("Cannot open video file")
    except Exception as e:
        print(f"Error opening video file: {e}")
        return

    threading.Thread(target=heavy_analysis_worker, args=(video_clip,), daemon=True).start()
    cv2.namedWindow('Lie Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lie Detector', 1280, 720)
    frame_count, start_time = 0, time.time()
    
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('p'): paused = not paused
        if key in [ord('1'), ord('2'), ord('3')]: playback_speed = int(chr(key))
        if paused:
            time.sleep(0.1)
            continue
        ret, frame = cap.read()
        if not ret: break
        for _ in range(playback_speed - 1):
            cap.grab()
            frame_count += 1
        frame_count += 1
        
        is_calibrating = not calibration_complete
        if is_calibrating and time.time() - start_time > CALIBRATION_SECONDS:
            calibration_complete = True
            for key in baselines:
                if baselines[key]: calibrated_baselines[key] = int(np.median(baselines[key]))
            print(f"Calibration Complete. Baselines: {calibrated_baselines}")
            is_calibrating = False

        frame_small = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        face_results = mp_face_mesh.process(frame_rgb)
        hands_results = mp_hands.process(frame_rgb)
        local_results = {}

        if face_results.multi_face_landmarks:
            local_results["Face Status"] = "Face Detected"
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            ih, iw, _ = frame_small.shape
            forehead_y_start = int(face_landmarks[10].y * ih - 20)
            forehead_y_end = int(face_landmarks[10].y * ih)
            forehead_x_start = int(face_landmarks[234].x * iw)
            forehead_x_end = int(face_landmarks[454].x * iw)
            forehead_roi = frame_rgb[forehead_y_start:forehead_y_end, forehead_x_start:forehead_x_end]
            if forehead_roi.size > 0:
                rppg_buffer.append(np.mean(forehead_roi[:,:,1]))
            bpm, resp = calculate_rppg_vitals(rppg_buffer, fps)
            if bpm: local_results["Heart Rate (BPM)"] = bpm
            if resp: local_results["Respiration Rate"] = resp
            if is_calibrating and bpm: baselines["bpm"].append(bpm)
            
            left_ear = compute_EAR(face_landmarks, [33, 160, 158, 133, 153, 144], iw, ih)
            right_ear = compute_EAR(face_landmarks, [362, 385, 387, 263, 373, 380], iw, ih)
            if (left_ear + right_ear) / 2.0 < EAR_THRESHOLD:
                if not any(time.time() - t < 0.3 for t in blink_timestamps):
                    blink_timestamps.append(time.time())
            
            one_minute_ago = time.time() - 60
            while blink_timestamps and blink_timestamps[0] < one_minute_ago:
                blink_timestamps.popleft()
            local_results["Blink Rate"] = len(blink_timestamps)
            if is_calibrating: baselines["blink_rate"].append(len(blink_timestamps))

            local_results["Lip Compression"] = "Detected" if compute_MAR(face_landmarks, iw, ih) < MAR_COMPRESSION_THRESHOLD else "Normal"
            local_results["Eye Gaze"] = analyze_eye_gaze(face_landmarks)
            
            if frame_count % int(fps/2) == 0 and analysis_queue.qsize() < 2:
                current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                analysis_queue.put((frame_rgb, face_landmarks, hands_results.multi_hand_landmarks, current_time_sec))
        else:
            local_results["Face Status"] = "No Face Detected"

        with results_lock:
            analysis_results.update(local_results)
            if calibration_complete:
                analysis_results["Deception Likelihood"] = calculate_deception_likelihood(analysis_results, calibrated_baselines)

        display_frame = draw_overlay(frame_small.copy(), analysis_results, is_calibrating)
        cv2.imshow('Lie Detector', cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
