import os
import json
import urllib.parse
import re
from flask import Flask, render_template, request, jsonify, send_from_directory
from model import analyze

app = Flask(__name__)

# Load benchmark metadata
METADATA_PATH = os.path.join(os.path.dirname(__file__), "models", "metadata.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        url = data.get("url", "")
        message = data.get("message", "")
        result = analyze(url=url, message=message)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/url-info", methods=["GET"])
def url_info():
    raw = request.args.get("url", "").strip()
    if not raw:
        return jsonify({"error": "No URL provided"}), 400
    u = raw if re.match(r"^[a-zA-Z]+://", raw) else "http://" + raw
    p = urllib.parse.urlparse(u)
    host = p.netloc.split("@")[-1].split(":")[0]
    scheme = p.scheme.lower()
    path = p.path if p.path else "/"
    if p.query:
        path += "?" + p.query
    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))
    tld = host.split(".")[-1] if "." in host and not is_ip else ("IP" if is_ip else "N/A")
    return jsonify({
        "url": u,
        "scheme": scheme,
        "host": host,
        "path": path,
        "is_ip": is_ip,
        "is_https": scheme == "https",
        "tld": tld,
        "favicon": f"https://www.google.com/s2/favicons?domain={host}&sz=64" if host else ""
    })

@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Benchmark metadata not found"}), 404

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "PhishGuard AI Risk Engine",
        "version": "2.0.0"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
