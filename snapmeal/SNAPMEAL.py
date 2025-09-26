import os
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io
from flask_cors import CORS

# --- Configuration ---
app = Flask(__name__)
CORS(app) # Enable CORS for frontend development
# Absolute path to your ONNX model weights file (UPDATE THIS!)
MODEL_PATH = r"D:\college\S5 STUFFS\ML\ml project\SNAPMEAL\best (1).onnx" 

# Mock Nutrition Data (Replace with a real API call if needed)
MOCK_NUTRITION = {
    "apple": {"calories": 95, "protein": 0.5, "fat": 0.3, "carbs": 25, "unit": "per 1 medium"},
    "banana": {"calories": 105, "protein": 1.3, "fat": 0.4, "carbs": 27, "unit": "per 1 medium"},
    "sandwich": {"calories": 350, "protein": 15, "fat": 18, "carbs": 35, "unit": "per average serving"},
    "appam": {"calories": 138, "protein": 1.9, "fat": 6.1, "carbs": 19.5, "unit": "per 1 standard (50g)"},
}

# Load YOLO model once
try:
    yolo_model = YOLO(MODEL_PATH)
    print("YOLO Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}. Check MODEL_PATH.")

def get_nutrition_data(food_name):
    """Fetches mock nutrition data based on the detected food name."""
    name_lower = food_name.lower()
    return MOCK_NUTRITION.get(name_lower, {
        "calories": "N/A", "protein": "N/A", "fat": "N/A", "carbs": "N/A", 
        "unit": "Data requires API lookup",
        "note": "⚠️ Data is mocked or missing."
    })

# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML upload page."""
    # This requires a 'templates' folder with index.html inside
    return render_template('snapmeal.html')

@app.route('/detect', methods=['POST'])
def detect_food_and_nutrition():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image data
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Perform YOLO Inference
        results = yolo_model.predict(image, conf=0.60, save=False, verbose=False)
        result_obj = results[0]
        detected_classes_map = result_obj.names
        boxes = result_obj.boxes

        detected_foods_data = []
        total_nutrition = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}

        # Process Detections and Fetch Nutrition
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = detected_classes_map[cls_id]

            nutrition = get_nutrition_data(class_name) 

            # Tally total nutrition
            try:
                total_nutrition["calories"] += float(nutrition.get("calories", 0))
                total_nutrition["protein"] += float(nutrition.get("protein", 0))
                total_nutrition["fat"] += float(nutrition.get("fat", 0))
                total_nutrition["carbs"] += float(nutrition.get("carbs", 0))
            except ValueError:
                pass # Skip if nutrition data is "N/A"

            detected_foods_data.append({
                "food_name": class_name,
                "confidence": f"{conf:.2f}",
                "nutrition": nutrition
            })
        
        # Return the results as JSON
        return jsonify({
            "status": "success",
            "detections": detected_foods_data,
            "total_nutrition": {k: f"{v:.2f}" for k, v in total_nutrition.items()}
        })

    except Exception as e:
        print(f"An error occurred during detection: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Flask will serve on http://127.0.0.1:5000/
    app.run(debug=True, port=5000)