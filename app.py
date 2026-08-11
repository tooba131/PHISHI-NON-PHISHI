from flask import Flask, jsonify, render_template_string, request

from phishing_detector import predict_url

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhishiLink Detector</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f7f9fc; color: #1f2937; }
    .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
    input[type="text"] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #d1d5db; margin-top: 10px; box-sizing: border-box; }
    button { margin-top: 12px; padding: 10px 16px; border: none; cursor: pointer; border-radius: 8px; background: #2563eb; color: white; }
    .result { margin-top: 16px; padding: 14px; border-radius: 8px; }
    .phish { background: #fee2e2; color: #991b1b; }
    .safe { background: #dcfce7; color: #166534; }
    pre { white-space: pre-wrap; background: #f3f4f6; padding: 12px; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>PhishiLink Detector</h1>
    <p>Paste a URL to see whether the trained model flags it as phishing.</p>
    <input id="url" type="text" placeholder="https://example.com" />
    <button onclick="checkUrl()">Check URL</button>
    <div id="result"></div>
  </div>
  <script>
    async function checkUrl() {
      const url = document.getElementById('url').value;
      const result = document.getElementById('result');
      result.innerHTML = 'Checking...';
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      const data = await response.json();
      const cls = data.is_phishing ? 'phish' : 'safe';
      result.className = 'result ' + cls;
      result.innerHTML = `<strong>${data.label.toUpperCase()}</strong><br>Probability: ${(data.probability * 100).toFixed(2)}%<br><pre>${data.char_stream}</pre>`;
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please provide a URL"}), 400
    return jsonify(predict_url(url))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
