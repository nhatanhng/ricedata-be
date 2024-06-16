# import all libraires
import logging

from io import BytesIO
from flask import Flask,  request, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from models import db, Files

# Initialize flask  and create sqlite database
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'nhatanhng'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ricedata.db'


SQLALCHEMY_TRACK_MODIFICATIONS = False
# SQLALCHEMY_ECHO = True
# db = SQLAlchemy(app)
db.init_app(app) 


# Configure logging
logging.basicConfig(level=logging.DEBUG)

with app.app_context():
    db.create_all()


@app.route('/uploads/images', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        upload = Files(filename=file.filename, data=file.read())
        db.session.add(upload)
        db.session.commit()
        return f'Uploaded: {file.filename}'
    
def get_file():
    if request.method == 'GET':
        files = Files.query.all()
        return jsonify([{'id': file.id, 'filename': file.filename} for file in files]), 200

 
# Create download function for download files
@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    try:
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404
        return send_file(BytesIO(file.data), download_name=file.filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404
        db.session.delete(file)
        db.session.commit()
        return jsonify({"message": f"File {filename} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)