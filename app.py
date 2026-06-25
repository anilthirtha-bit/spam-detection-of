from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained spam model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Spam Detection API"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "message field required"}), 400

        message = data["message"]

        prediction = model.predict([message])[0]

        return jsonify({
            "message": message,
            "spam": bool(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
