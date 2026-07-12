import time
import cv2
import requests
from crypto_utils import DERIVED_KEY, encrypt_bytes

SERVER_URL = "http://127.0.0.1:8000/explore"

print("Opening webcam for scene exploration snapshot...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

time.sleep(2)  # Adjust sensor exposure adjustments

print("Capturing full high-resolution snapshot...")
ret, frame = cap.read()
cap.release()
print("Webcam released.")

if not ret:
    print("Error: Failed to capture scene snapshot.")
    exit()

# Keep resolution high for VLM clarity, encode to JPEG buffer
success, encoded_image = cv2.imencode(".jpg", frame)

if success:
    raw_jpg_bytes = encoded_image.tobytes()
    
    print("Encrypting exploration snapshot with AES-GCM-SIV...")
    encrypted_payload = encrypt_bytes(raw_jpg_bytes, DERIVED_KEY)

    # Set up user-customized inspection prompt question
    user_question = "What objects are on the desk, and is there any text visible?"
    
    # Pass prompt inside custom header alongside streaming binary declaration 
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Explore-Prompt": user_question
    }

    print("Transmitting encrypted snapshot payload to vision network...")
    start_time = time.time()
    
    response = requests.post(SERVER_URL, data=encrypted_payload, headers=headers)
    
    rtt = round((time.time() - start_time) * 1000, 2)

    if response.status_code == 200:
        data = response.json()
        print("\n--- SECURE EXPLORATION DESCRIPTION RESPONSE ---")
        print(f"Total Turnaround: {rtt} ms ({round(rtt / 1000, 2)} seconds)")

        if data.get("status") == "success":
            print(f"Question Processed: '{data.get('prompt_used')}'")
            print(f"\nSpeech Engine Output:\n{data.get('speech_output')}")
        else:
            print(f"Server Processing Error: {data.get('message')}")
        print("-------------------------------------------------\n")
    else:
        print(f"HTTP Target Connection Failure: {response.status_code} - {response.text}")
