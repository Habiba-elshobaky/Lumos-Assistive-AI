import cv2
from ultralytics import YOLO
import speech_recognition as sr
import threading
from gtts import gTTS
import pygame
import winsound
import os
import time
import brain_module as luma_brain

# --- 1. AUDIO & VOICE CONFIGURATION ---
pygame.mixer.init()

def speak(text, emergency=False):
    """ 
    Converts text to speech. If emergency=True, it kills current 
    audio to prioritize danger warnings. 
    """
    try:
        if text.strip():
            # Stop any ongoing speech if a danger is detected
            if emergency:
                pygame.mixer.music.stop()
            
            print(f">>> [LUMA]: {text}")
            tts = gTTS(text=text, lang='en')
            filename = "luma_temp_voice.mp3"
            tts.save(filename)
            
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Keep loop active until speech is done
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
    except Exception as e:
        print(f">>> [VOICE ERROR]: {e}")

def play_alert(freq=1000):
    """ Plays a beep sound with a specific frequency. """
    winsound.Beep(freq, 200)

# --- 2. OBJECT DETECTION SETTINGS ---
FOCAL_LENGTH = 160
REAL_HEIGHT_PERSON = 170

# Normal obstacles that trigger "Caution"
TRIP_HAZARDS = ['chair', 'bench', 'potted plant', 'suitcase', 'box', 'person']
# High-speed danger objects that trigger "Emergency Override"
DANGER_OBJECTS = ['car', 'truck', 'bus', 'motorcycle']

last_safety_alert = ""
pending_speech = "" 
emergency_active = False # Flag to disable other tasks during danger

# --- 3. VOICE ASSISTANT LISTENER (THREADED) ---
def listen_thread(frame_copy):
    """ Listens to the user and sends the query to the Gemini Brain. """
    global pending_speech, emergency_active
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300 
    
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            play_alert(1200) # Start beep
            print(">>> [LUMA]: Listening...")
            
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            command = recognizer.recognize_google(audio).lower()
            print(f">>> [USER]: {command}")

            # Do not process AI requests if there is a life-threatening danger nearby
            if not emergency_active:
                cv2.imwrite("temp_scene.jpg", frame_copy)
                result = luma_brain.describe_scene("temp_scene.jpg", command)
                pending_speech = result 

        except Exception as e:
            print(f">>> [MIC INFO]: {e}")
            if not emergency_active:
                pending_speech = "I couldn't hear you clearly."

# --- 4. MAIN EXECUTION LOOP ---
model = YOLO('yolo11n.pt') 
cap = cv2.VideoCapture(0)

print("\n--- LUMOS MULTI-MODE ASSISTANT ACTIVE ---")
print("MODES: [Walking Mode (Always On) | Social & Find Mode (Space Bar)]\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    key = cv2.waitKey(1) & 0xFF
    
    # 4.1 EMERGENCY OVERRIDE (PRIORITY 1)
    results = list(model.predict(frame, conf=0.35, stream=True, verbose=False))
    current_emergency = False

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0]
            dist = (REAL_HEIGHT_PERSON * FOCAL_LENGTH) / ((y2 - y1) * 100)
            
            # If a car is closer than 4 meters, trigger EMERGENCY
            if label in DANGER_OBJECTS and dist < 4.0: 
                current_emergency = True
                emergency_active = True
                pending_speech = "" # Clear any pending AI responses
                play_alert(2500) # High-pitched alarm
                speak(f"DANGER! {label} approaching!", emergency=True)
                break 

    # If emergency is detected, skip all other processing for this frame
    if current_emergency:
        cv2.imshow("LUMOS FINAL BUILD", frame)
        if key == ord('q'): break
        continue 

    emergency_active = False # Reset flag if path is clear

    # 4.2 VOICE ASSISTANT TRIGGER (PRIORITY 2)
    if key == ord(' ') or key == ord('v'):
        threading.Thread(target=listen_thread, args=(frame.copy(),), daemon=True).start()

    # 4.3 PROCESS PENDING AI SPEECH (SOCIAL/FIND MODE)
    if pending_speech != "":
        text_to_say = pending_speech
        pending_speech = "" 
        speak(text_to_say)

    # 4.4 NORMAL WALKING SAFETY (PRIORITY 3)
    for r in results:
        annotated_frame = r.plot()
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0]
            dist = (REAL_HEIGHT_PERSON * FOCAL_LENGTH) / ((y2 - y1) * 100)
            center_x = (x1 + x2) / 2
            
            # Normal hazard detection (Center of view, under 1.6 meters)
            if label in TRIP_HAZARDS and dist < 1.6 and (w/3 < center_x < 2*w/3):
                if label != last_safety_alert:
                    play_alert(800)
                    threading.Thread(target=speak, args=(f"Caution, {label}",)).start()
                    last_safety_alert = label
            elif dist > 2.5: 
                last_safety_alert = ""

    cv2.imshow("LUMOS FINAL BUILD", annotated_frame)
    if key == ord('q'): break 

cap.release()
cv2.destroyAllWindows()