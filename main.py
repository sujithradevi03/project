import cv2
import pyttsx3
import threading
import queue
import time
import speech_recognition as sr
from ultralytics import YOLO

# --- TEXT TO SPEECH SETUP ---
# Initialize TTS engine in a separate thread to prevent blocking
class TTSThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.engine = None

    def run(self):
        print("TTS Thread: Started.")
        while True:
            text = self.queue.get()
            if text is None:
                break
            
            print(f"TTS Thread: Processing -> '{text}'")
            try:
                # Create a fresh engine instance for each message
                # This avoids long-lived COM object issues in threads
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                del engine
                print("TTS Thread: Finished speaking.")
            except Exception as e:
                print(f"TTS Thread Error: {e}")
            finally:
                self.queue.task_done()

# Initialize Queue and TTS Thread
speech_queue = queue.Queue()

# Start TTS Thread
tts_thread = TTSThread(speech_queue)
tts_thread.start()

# Helper function
def speak(text):
    # Avoid filling queue with too many messages
    if speech_queue.qsize() < 2:
        speech_queue.put(text)

# Queue startup message
time.sleep(1) # Give thread a moment to init
speech_queue.put("System online. Waiting for command.")

# --- SPEECH RECOGNITION SETUP ---
# Global flag to trigger capture
capture_requested = False
listening_status = "Initializing..."

def command_listener():
    global capture_requested, listening_status
    recognizer = sr.Recognizer()
    
    # Check for microphone
    try:
        with sr.Microphone() as source:
            print("Microphone detected.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            listening_status = "Listening for 'Yes'..."
            
            while True:
                try:
                    # Listen for audio (timeout to allow loop to check for exit, but we are daemon)
                    # print("Listening...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    
                    try:
                        text = recognizer.recognize_google(audio).lower()
                        print(f"Heard: {text}")
                        
                        if "yes" in text or "capture" in text or "scan" in text:
                            print("Command 'YES' detected.")
                            capture_requested = True
                            speak("Okay, analyzing scene.")
                        
                    except sr.UnknownValueError:
                        pass # Could not understand audio
                    except sr.RequestError as e:
                        print(f"Could not request results; {0}".format(e))
                        
                except sr.WaitTimeoutError:
                    pass # Just loop
                except Exception as e:
                    print(f"Listener Error: {e}")
                    time.sleep(1)
                    
    except OSError:
        listening_status = "No Microphone Found"
        print("Error: No microphone found.")

# Start listener thread
listener_thread = threading.Thread(target=command_listener, daemon=True)
listener_thread.start()


# --- SPATIAL LOGIC ---
def get_spatial_description(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    
    # Horizontal Position
    if center_x < frame_width / 3:
        h_pos = "on the left"
    elif center_x > 2 * frame_width / 3:
        h_pos = "on the right"
    else:
        h_pos = "in the center"
        
    # Vertical/Proximity
    frame_area = frame_width * frame_height
    box_area = (x2 - x1) * (y2 - y1)
    ratio = box_area / frame_area
    
    if ratio > 0.15:
        proximity = "near"
    elif ratio < 0.05:
        proximity = "far"
    else:
        proximity = "at a medium distance"
        
    return f"{h_pos}, {proximity}"

def main():
    global capture_requested, listening_status
    
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    try:
        model = YOLO("yolov8n.pt")  # NANO model
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("System Ready. Say 'Yes' to capture.")
    speak("Camera is ready. Please say yes to capture the scene.")

    last_analysis_text = "Waiting for command..."

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            height, width, _ = frame.shape
            
            # Use current frame for display
            display_frame = frame.copy()
            
            # --- PROCESS COMMAND ---
            if capture_requested:
                capture_requested = False
                # Run inference on current 'frame'
                results = model(frame, verbose=False)
                
                detections = []
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r = box.xyxy[0].astype(int)
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        label = result.names[cls]
                        
                        if conf > 0.5:
                            spatial_info = get_spatial_description(r, width, height)
                            detections.append(f"{label} {spatial_info}")
                            
                            # Draw on display frame for visual confirmation
                            cv2.rectangle(display_frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{label} {conf:.2f}", (r[0], r[1] - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Generate Caption
                if detections:
                    # Construct description
                    # "I see a person on the left, near. And a cup on the right."
                    main_desc = f"I found {len(detections)} objects. "
                    # Limit to 3 items to be concise
                    items_desc = ". ".join(detections[:3])
                    full_caption = main_desc + items_desc
                    speak(full_caption)
                    last_analysis_text = full_caption
                else:
                    speak("I analyzed the scene but did not detect any clear objects.")
                    last_analysis_text = "No objects found."

            # UI Overlay
            cv2.putText(display_frame, f"Status: {listening_status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show last analysis result at bottom
            # Wrap text if too long
            cv2.putText(display_frame, last_analysis_text[:80], (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Context-Aware Image Captioning", display_frame)

            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'): # Manual capture alternative
                capture_requested = True

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_queue.put(None)
        tts_thread.join()
        print("System stopped.")

if __name__ == "__main__":
    main()
