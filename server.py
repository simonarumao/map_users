from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Store received locations for Leaflet display
locations = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trigger', methods=['POST'])
def trigger_alert():
    global locations
    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude and longitude:
        # Add to locations list
        locations.append({"latitude": latitude, "longitude": longitude})
        return jsonify({"message": "Alert triggered!", "latitude": latitude, "longitude": longitude}),200
    else:
        return jsonify({"error": "Invalid data"}), 400

@app.route('/locations', methods=['GET'])
def get_locations():
    return jsonify(locations)

if __name__ == '__main__':
    app.run()
