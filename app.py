from flask import Flask, render_template, request
from src.nlp_project.pipeline.prediction import PredictionPipeline
import os
from fastapi.responses import Response

app = Flask(__name__)

@app.route("/train")
def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    prediction = ""

    if request.method == "POST":
        text = request.form.get("text")
        obj = PredictionPipeline()
        prediction = obj.predict(text)

    return render_template("predict.html", text=text, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
