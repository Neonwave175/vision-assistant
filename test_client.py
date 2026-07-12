import time
import cv2
import requests
from crypto_utils import DERIVED_KEY, encrypt_bytes

SERVER_URL = "http://127.0.0.1:8000/analyze"

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam ready. Capturing frame...")
time.sleep(2)  # Warm up sensor

ret, frame = cap.read()
cap.release()
print("Webcam closed.")

if not ret:
    print("Error: Failed to capture image.")
    exit()

# Resize for efficient edge transfer
height, width, _ = frame.shape
resized_frame = cv2.resize(frame, (640, int(640 * (height / width))))

# Encode to JPEG in-memory buffer
success, encoded_image = cv2.imencode(".jpg", resized_frame)

if success:
    raw_jpg_bytes = encoded_image.tobytes()
    
    print("Encrypting frame using Argon2 derived key + AES-GCM-SIV...")
    encrypted_payload = encrypt_bytes(raw_jpg_bytes, DERIVED_KEY)
    
    print("Sending encrypted payload to M1 Max server...")
    start_time = time.time()
    
    # Send as raw binary data via data= instead of files=
    headers = {"Content-Type": "application/octet-stream"}
    response = requests.post(SERVER_URL, data=encrypted_payload, headers=headers)
    
    rtt = round((time.time() - start_time) * 1000, 2)

    if response.status_code == 200:
        data = response.json()
        print("\n--- SECURE SERVER RESPONSE ---")
        print(f"Round-Trip Time: {rtt} ms")
        print(f"Speech Output:   '{data['speech_output']}'")
        print(f"Detections:      {data['raw_detections']}")
        print("-------------------------------\n")
    else:
        print(f"Server Error: {response.status_code} - {response.text}")
