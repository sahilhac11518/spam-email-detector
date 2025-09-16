from flask import Flask, request, jsonify, render_template
# # Loading
# with open("spam_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)
# import joblib

from flask import Flask, request, jsonify, render_template
import joblib   # <-- add this

# Load model
model = joblib.load("spam_model.pkl")

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("spam_email.html")  # HTML in templates folder

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email_text = data.get("email", "")

    if not email_text.strip():
        return jsonify({"error": "Empty email text"}), 400

    features = vectorizer.transform([email_text])
    prediction = model.predict(features)[0]

    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
