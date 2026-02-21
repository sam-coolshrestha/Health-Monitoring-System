from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("health_risk_rf_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Get values from form
        respiratory_rate = float(request.form["respiratory_rate"])
        oxygen_saturation = float(request.form["oxygen_saturation"])
        o2_scale = int(request.form["o2_scale"])
        systolic_bp = float(request.form["systolic_bp"])
        heart_rate = float(request.form["heart_rate"])
        temperature = float(request.form["temperature"])
        consciousness = int(request.form["consciousness"])
        on_oxygen = int(request.form["on_oxygen"])

        # Arrange in same order as training
        input_data = np.array([[respiratory_rate, oxygen_saturation, o2_scale,
                                systolic_bp, heart_rate, temperature,
                                consciousness, on_oxygen]])

        result = model.predict(input_data)[0]

        risk_map = {
            0: "Normal",
            1: "Low Risk",
            2: "Medium Risk",
            3: "High Risk"
        }

        prediction = risk_map[result]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)