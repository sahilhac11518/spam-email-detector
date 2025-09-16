from flask import Flask, request, jsonify, render_template
import joblib

# Load the entire model pipeline (including vectorizer + classifier)
model = joblib.load("spam_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("spam_email.html")  # HTML in 'templates/' folder

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email_text = data.get("email", "")

    if not email_text.strip():
        return jsonify({"error": "Empty email text"}), 400

    # Predict using the model pipeline
    prediction = model.predict([email_text])  # wrap input in list

    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
