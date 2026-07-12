import base64
import io
from fastapi import FastAPI, File, Form, UploadFile
from openai import OpenAI
from PIL import Image
import torch
from ultralytics import YOLO

# 1. Initialize FastAPI
app = FastAPI(title="Visual Assistant Edge Server")

# 2. Check for M1 Max Hardware Acceleration (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading YOLO on device: {device.upper()}")

# 3. Load YOLO Model for Loop 1 (Continuous Safety)
model = YOLO("yolo11n.pt")
model.to(device)

# Define Priority 1 Hazards (Immediate danger)
HAZARDS = {"car", "truck", "bus", "motorcycle", "bicycle", "dog"}
# Define Priority 2 Navigation Aids
NAV_AIDS = {"person", "traffic light", "stop sign", "bench", "fire hydrant"}

# 4. Initialize OpenAI client pointing to Local LM Studio Server
# LM Studio defaults to port 1234, api_key is not required but must be non-empty
lm_studio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    # Read the incoming image bytes
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img_width, img_height = img.size
    img_area = img_width * img_height

    # Run YOLO Inference
    results = model(img, verbose=False)[0]

    detections = []
    priority_1_alerts = []
    priority_2_alerts = []

    # Process each bounding box detected in the frame
    for box in results.boxes:
        conf = box.conf[0].item()
        if conf < 0.5:  # Skip low confidence detections
            continue

        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]

        # Get box coordinates [x_min, y_min, x_max, y_max]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Calculate relative position (Left, Center, Right)
        center_x = (x1 + x2) / 2
        if center_x < img_width / 3:
            direction = "on your left"
        elif center_x > (2 * img_width) / 3:
            direction = "on your right"
        else:
            direction = "straight ahead"

        # Calculate relative distance based on bounding box size
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / img_area

        if area_ratio > 0.25:
            distance = "very close"
        elif area_ratio > 0.08:
            distance = "approaching"
        else:
            distance = "in the distance"

        detections.append(
            {
                "label": label,
                "confidence": round(conf, 2),
                "direction": direction,
                "distance": distance,
            }
        )

        # Apply Priority Matrix Filtering
        if label in HAZARDS and distance in ["very close", "approaching"]:
            priority_1_alerts.append(f"Caution: {label} {distance} {direction}.")
        elif label in NAV_AIDS and distance != "in the distance":
            priority_2_alerts.append(f"{label} {direction}.")

    # Decide what text string to send back to the speaker
    if priority_1_alerts:
        # Hazards interrupt and override everything else
        final_speech = " ".join(priority_1_alerts[:2])  # Max 2 alerts to stay brief
    elif priority_2_alerts:
        # If no hazards, speak navigation aids
        final_speech = " ".join(priority_2_alerts[:2])
    else:
        # Priority 4: Background clutter is ignored
        final_speech = "Path looks clear."

    return {
        "status": "success",
        "speech_output": final_speech,
        "raw_detections": detections,
    }

# --- LOOP 2: ON-DEMAND EXPLORATION ENDPOINT (LM STUDIO VERSION) ---
@app.post("/explore")
async def explore_scene(
    file: UploadFile = File(...),
    prompt: str = Form(
        "Describe this scene for a visually impaired person. Be concise, mention key objects, their layout, and any readable text."
    ),
):
    image_bytes = await file.read()

    try:
        # Convert image bytes to a Base64 data URI string for LM Studio
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        # Send request using the standard OpenAI Vision protocol
        response = lm_studio_client.chat.completions.create(
            model="meta-llama-3.2-11b-vision-instruct",  # LM Studio auto-routes to whatever model you have loaded
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

        # Extract the text answer
        description = response.choices[0].message.content

        return {
            "status": "success",
            "prompt_used": prompt,
            "speech_output": description,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"LM Studio execution failed: {str(e)}. Make sure LM Studio Local Server is running on port 1234.",
        }