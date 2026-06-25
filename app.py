from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return {"status": "Spam Detection API Running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "message field required"}), 400

    message = data["message"]

    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]

    result = "spam" if prediction == 1 else "not spam"

    return jsonify({
        "message": message,
        "prediction": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
