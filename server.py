import io
import base64
from fastapi import FastAPI, Request, HTTPException, Form
from openai import OpenAI
from PIL import Image
import torch
from ultralytics import YOLO

# Import the pre-derived key and decryption routine
from crypto_utils import DERIVED_KEY, decrypt_bytes

app = FastAPI(title="Encrypted Visual Assistant Edge Server")

# Check for Hardware Acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading YOLO on device: {device.upper()}")

model = YOLO("yolo11n.pt")
model.to(device)

HAZARDS = {"car", "truck", "bus", "motorcycle", "bicycle", "dog"}
NAV_AIDS = {"person", "traffic light", "stop sign", "bench", "fire hydrant"}

lm_studio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

@app.post("/analyze")
async def analyze_frame(request: Request):
    # Read raw body bytes directly from the stream
    encrypted_bytes = await request.body()
    
    try:
        # Decrypt payload back into raw JPEG bytes
        plaintext_bytes = decrypt_bytes(encrypted_bytes, DERIVED_KEY)
        img = Image.open(io.BytesIO(plaintext_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption or image parsing failed: {str(e)}")

    img_width, img_height = img.size
    img_area = img_width * img_height

    results = model(img, verbose=False)[0]
    detections = []
    priority_1_alerts = []
    priority_2_alerts = []

    for box in results.boxes:
        conf = box.conf[0].item()
        if conf < 0.5:
            continue

        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Spatial Position Calculations
        center_x = (x1 + x2) / 2
        if center_x < img_width / 3:
            direction = "on your left"
        elif center_x > (2 * img_width) / 3:
            direction = "on your right"
        else:
            direction = "straight ahead"

        # Proximity Estimation
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / img_area

        if area_ratio > 0.25:
            distance = "very close"
        elif area_ratio > 0.08:
            distance = "approaching"
        else:
            distance = "in the distance"

        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "direction": direction,
            "distance": distance,
        })

        if label in HAZARDS and distance in ["very close", "approaching"]:
            priority_1_alerts.append(f"Caution: {label} {distance} {direction}.")
        elif label in NAV_AIDS and distance != "in the distance":
            priority_2_alerts.append(f"{label} {direction}.")

    if priority_1_alerts:
        final_speech = " ".join(priority_1_alerts[:2])
    elif priority_2_alerts:
        final_speech = " ".join(priority_2_alerts[:2])
    else:
        final_speech = "Path looks clear."

    return {
        "status": "success",
        "speech_output": final_speech,
        "raw_detections": detections,
    }


@app.post("/explore")
async def explore_scene(request: Request):
    """
    To pass fields alongside raw binary payloads, query parameters or custom headers 
    can be parsed here. Expecting raw data body to contain the encrypted image.
    """
    encrypted_bytes = await request.body()
    
    # Extract optional prompt from header or fallback
    prompt = request.headers.get("X-Explore-Prompt", "Describe this scene concisely.")

    try:
        plaintext_bytes = decrypt_bytes(encrypted_bytes, DERIVED_KEY)
        base64_image = base64.b64encode(plaintext_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        response = lm_studio_client.chat.completions.create(
            model="meta-llama-3.2-11b-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            max_tokens=300,
        )

        return {
            "status": "success",
            "prompt_used": prompt,
            "speech_output": response.choices[0].message.content,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exploration routine failed: {str(e)}")
