import time
import cv2
import requests

# Point to the new /explore endpoint
SERVER_URL = "http://127.0.0.1:8000/explore"

print("Opening Mac Webcam for exploration snapshot...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

time.sleep(2)  # Give webcam sensor time to adjust to light

print("Capturing high-resolution snapshot...")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Failed to capture image.")
    exit()

# Encode frame to JPEG
success, encoded_image = cv2.imencode(".jpg", frame)

if success:
    print("Sending snapshot to Ollama VLM on M1 Max...")

    # Package image file
    files = {"file": ("explore_frame.jpg", encoded_image.tobytes(), "image/jpeg")}

    # Define what the user wants to know (you can change this string!)
    user_question = "What objects are on the desk, and is there any text visible?"
    payload = {"prompt": user_question}

    start_time = time.time()
    # Send both image file and form data prompt
    response = requests.post(SERVER_URL, files=files, data=payload)
    rtt = round((time.time() - start_time) * 1000, 2)

    if response.status_code == 200:
        data = response.json()
        print("\n--- SERVER RESPONSE ---")
        print(f"Round-Trip Time: {rtt} ms ({round(rtt / 1000, 2)} seconds)")

        if data.get("status") == "success":
            print(f"Question Asked:  '{data.get('prompt_used')}'")
            print(f"\nSpeech Output:\n{data.get('speech_output')}")
        else:
            print(f"Server Internal Error: {data.get('message')}")

        print("----------------------------\n")
    else:
        print(f"HTTP Connection Error: {response.status_code} - {response.text}")