from flask import Flask, render_template, request

from analyzer import analyze_case


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    # Read water intake from the HTML form
    water = float(request.form.get("water", 0) or 0)

    case_data = {
        "Age": request.form.get("age"),
        "Sleep hours per day": request.form.get("sleep"),
        "Exercise minutes per week": request.form.get("exercise"),
        "Stress level": request.form.get("stress"),
        "Water intake per day": water,
        "Smoking status": request.form.get("smoking"),
        "Fruit and vegetable intake": request.form.get("diet"),
        "Persistent fatigue": request.form.get("fatigue"),
        "Additional notes": request.form.get("notes")
    }

    result = analyze_case(case_data)

    return render_template(
        "result.html",
        result=result
    )


if __name__ == "__main__":
    app.run()