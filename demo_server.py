from flask import Flask, send_from_directory, request, jsonify
import os
import re
import test_video

app = Flask(__name__)
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_pages')

@app.route('/')
def index():
    return send_from_directory(DEMO_DIR, 'index.html')

@app.route('/scan-video', methods=['POST'])
def scan_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = os.path.join(DEMO_DIR, '_scanned_temp.mp4')
    file.save(temp_path)

    try:
        report = test_video.run_forensic_analysis(temp_path)
        report['video_url'] = '/_scanned_temp.mp4'
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:filename>')
def serve_file(filename):
    path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(path):
        from flask import abort
        abort(404)
        
    from flask import send_file
    return send_file(path, conditional=True)

@app.after_request
def add_header(response):
    response.headers['Accept-Ranges'] = 'bytes'
    return response

if __name__ == '__main__':
    print("=" * 60)
    print("Starting PhishGuard Demo & Scanner Server on http://127.0.0.1:8000")
    print("Web Deepfake Scanner: http://127.0.0.1:8000/deepfake_scanner.html")
    print("=" * 60)
    app.run(port=8000, debug=False)
