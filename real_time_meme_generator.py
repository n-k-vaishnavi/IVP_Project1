import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from pathlib import Path

# --- Step 1: Capture image from webcam ---
cam = cv2.VideoCapture(0)
print("Press SPACE to take a photo, ESC to exit.")

while True:
    ret, frame = cam.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    
    if key % 256 == 27:  # ESC
        cam.release()
        cv2.destroyAllWindows()
        exit()
    elif key % 256 == 32:  # SPACE
        img_name = "captured_photo.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Photo saved as {img_name}")
        break

cam.release()
cv2.destroyAllWindows()

# --- Step 2: Load image for AI captioning ---
raw_image = Image.open(img_name).convert("RGB")

# --- Step 3: Load BLIP model ---
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --- Step 4: Generate caption ---
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("AI-generated caption:", caption)

# --- Step 5: Open image with Pillow for text overlay ---
img = Image.open(img_name)
draw = ImageDraw.Draw(img)

# --- Step 6: Load Anton font ---
font_path = Path(__file__).parent / "Anton-Regular.ttf"
font = ImageFont.truetype(str(font_path), 50)

# --- Step 7: Center caption at bottom ---
img_width, img_height = img.size
bbox = draw.textbbox((0,0), caption, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (img_width - text_width)/2
y = img_height - text_height - 20

# --- Step 8: Draw black outline ---
outline_range = 2
for ox in range(-outline_range, outline_range+1):
    for oy in range(-outline_range, outline_range+1):
        draw.text((x+ox, y+oy), caption, font=font, fill="black")

# --- Step 9: Draw white text ---
draw.text((x, y), caption, font=font, fill="white")

# --- Step 10: Save final meme ---
meme_name = "smart_meme.jpg"
img.save(meme_name)
print(f"Smart meme saved as {meme_name}")
