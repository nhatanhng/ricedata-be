# import all libraries
import logging
import os
from flask import Flask, request, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models import db, Files
from io import BytesIO

# Initialize Flask and create SQLite database
app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'nhatanhng'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ricedata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

with app.app_context():
    db.create_all()

@app.route('/uploads/images', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            upload = Files(filename=filename, filepath=filepath)
            db.session.add(upload)
            db.session.commit()
            return f'Uploaded: {filename}'
        else:
            return 'No file uploaded', 400

@app.route('/files', methods=['GET'])
def get_files():
    if request.method == 'GET':
        files = Files.query.all()
        return jsonify([{'id': file.id, 'filename': file.filename} for file in files]), 200

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    try:
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404
        return send_file(file.filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404
        os.remove(file.filepath)
        db.session.delete(file)
        db.session.commit()
        return jsonify({"message": f"File {filename} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rename/<filename>', methods=['PUT'])
def rename_file(filename):
    try:
        new_filename = request.json.get('newFilename')
        if not new_filename:
            return jsonify({"error": "New filename not provided"}), 400
        
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404

        new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(new_filename))
        os.rename(file.filepath, new_filepath)

        file.filename = new_filename
        file.filepath = new_filepath
        db.session.commit()
        return jsonify({"message": f"File {filename} renamed to {new_filename}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)