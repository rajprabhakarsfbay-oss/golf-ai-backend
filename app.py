# =============================================================================
# SAN RAMON GOLF AI - BACKEND SERVER (app.py) - STABILITY FIX
# =============================================================================
from flask import Flask, request, jsonify
from flask_cors import CORS 
import sqlite3
import json
import re
from datetime import datetime
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import requests 

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Constants ---
DATABASE = 'golf_ai.db'
MODEL_PATH = './san_ramon_golf_ai_50k_model'
WEATHER_API_KEY = 'cb0250ab8965651b4fc61881a300c013'
SAN_RAMON_LAT = 37.7799
SAN_RAMON_LON = -121.9780

# --- Global variables for the model ---
model = None
tokenizer = None

# --- Model Loading ---
def load_trained_model():
    """Loads the trained GPT-2 model and tokenizer."""
    global model, tokenizer
    try:
        logger.info(f"Loading trained model from {MODEL_PATH}")
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        logger.info("✅ Trained model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load trained model: {e}")
        return False

# --- Database Initialization ---
def init_database():
    """Initializes the SQLite database."""
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS golf_shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                hole_number INTEGER, distance_to_pin INTEGER, club_used TEXT,
                ball_flight TEXT, weather_conditions TEXT, lie_type TEXT,
                ai_recommendation TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    logger.info("✅ Database initialized")

# --- AI and Helper Functions ---
def get_live_weather():
    """Fetches live weather data from the server side."""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={SAN_RAMON_LAT}&lon={SAN_RAMON_LON}&appid={WEATHER_API_KEY}&units=imperial"
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info("✅ Successfully fetched live weather data.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching weather from API: {e}")
        return None

def extract_club_from_ai_text(ai_text):
    """Extracts a club name from the model's text output."""
    text_lower = ai_text.lower()
    match = re.search(r'\b(\d+[\s-]?iron|pw|sw|driver|putter|\d+[\s-]?wood)\b', text_lower)
    if match:
        club = match.group(1).replace(' ', '-')
        logger.info(f"🎯 AI recommended: {club}")
        return club
    logger.warning(f"⚠️ Could not extract club from AI text: '{ai_text[:50]}...'")
    return "N/A"

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.after_request
def after_request(response):
    """Manually add CORS headers to every response to ensure connectivity."""
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, ngrok-skip-browser-warning'
    header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/api/gps-recommendation', methods=['POST', 'OPTIONS'])
def gps_recommendation():
    """Main endpoint for getting AI recommendations and live weather in one call."""
    logger.info(f"Received request on /api/gps-recommendation from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204

    # *** NEW STABILITY CHECK ***
    # Verify that the model is loaded before proceeding.
    if not model or not tokenizer:
        logger.error("❌ Model not loaded. Cannot generate recommendation.")
        return jsonify({"error": "AI model is not available. Check server logs."}), 503

    try:
        live_weather = get_live_weather()
        if not live_weather:
            return jsonify({"error": "Failed to fetch up-to-the-minute weather data."}), 500

        data = request.json
        distance = data.get('distance')
        lie = data.get('lie')
        hole = data.get('hole')
        
        prompt = f"""San Ramon Golf Course Hole {hole} Recommendation:
        Distance: {distance} yards. Lie: {lie}.
        Wind: {live_weather['wind']['speed']:.1f}mph from {live_weather['wind']['deg']} degrees.
        Temp: {live_weather['main']['temp']:.1f}°F.
        Recommend club:"""
        
        logger.info(f"🧠 Asking trained AI: '{prompt}'")

        inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=100, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_length=inputs.shape[1] + 40, num_return_sequences=1,
                temperature=0.4, pad_token_id=tokenizer.eos_token_id, do_sample=True
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        insight = full_response[len(prompt):].strip().split('.')[0] + '.'

        response_data = {
            "AI_recommended": extract_club_from_ai_text(insight),
            "recommendation": insight,
            "plays_like": distance, # Placeholder
            "live_weather": live_weather 
        }
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"GPS recommendation endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """A simple endpoint to check if the server is running."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == '__main__':
    print("🏌️ Golf AI Backend starting...")
    init_database()
    
    if not load_trained_model():
        print("⚠️ WARNING: Trained model failed to load. AI recommendations will be unavailable.")
    
    print("🚀 Golf AI Backend is running!")
    print(f"🔗 Frontend should connect via Render URL.")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
