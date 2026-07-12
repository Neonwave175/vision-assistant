import time
import cv2
import requests

# Server URL (running locally on your Mac)
SERVER_URL = "http://127.0.0.1:8000/analyze"

print("Opening Mac Webcam...")
# 0 is usually the built-in Mac webcam. If you have an external cam, try 1.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam opened! Capturing a test frame in 2 seconds...")
time.sleep(2)  # Give webcam sensor time to adjust to room lighting

# Capture a single frame
ret, frame = cap.read()

if not ret:
    print("Error: Failed to capture image.")
    cap.release()
    exit()

# Resize frame slightly to simulate what the Raspberry Pi will send (faster transfer)
height, width, _ = frame.shape
resized_frame = cv2.resize(frame, (640, int(640 * (height / width))))

# Encode the frame into a JPEG memory buffer
success, encoded_image = cv2.imencode(".jpg", resized_frame)

if success:
    print("Sending frame to M1 Max server for analysis...")

    # Package the bytes as a file upload
    files = {"file": ("webcam_frame.jpg", encoded_image.tobytes(), "image/jpeg")}

    # Send POST request to FastAPI
    start_time = time.time()
    response = requests.post(SERVER_URL, files=files)
    rtt = round((time.time() - start_time) * 1000, 2)

    if response.status_code == 200:
        data = response.json()
        print("\n--- SERVER RESPONSE ---")
        print(f"Round-Trip Time: {rtt} ms")
        print(f"Speech Output:   '{data['speech_output']}'")
        print(f"Detections:      {data['raw_detections']}")
        print("-----------------------\n")
    else:
        print(f"Server Error: {response.status_code} - {response.text}")

# Release the webcam
cap.release()
print("Webcam closed.")