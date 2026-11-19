from flask import Flask, jsonify, request, send_from_directory

from .predictor import Predictor

app = Flask(__name__, static_folder="static", static_url_path="/static")
predictor = Predictor()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/schema")
def schema():
    return jsonify(predictor.schema())


@app.get("/health")
def health():
    ready = predictor.pipeline is not None
    return jsonify({"status": "ok" if ready else "model_unavailable"}), (200 if ready else 503)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    values = payload.get("features")
    if not isinstance(values, dict):
        return jsonify({"error": "validation_failed", "details": {"features": "object wajib diisi"}}), 400
    try:
        return jsonify(predictor.predict(values))
    except ValueError as exc:
        return jsonify({"error": "validation_failed", "details": exc.args[0]}), 400
    except RuntimeError as exc:
        return jsonify({"error": "model_unavailable", "message": str(exc)}), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
