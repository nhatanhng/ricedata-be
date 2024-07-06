import logging
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models import db, Files
from PIL import Image
from spectral import open_image
from models import db, Files
import io
import numpy as np

# Initialize Flask and create SQLite database
app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'nhatanhng'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ricedata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Directory to save uploaded files and visualized images
UPLOAD_FOLDER = 'uploads'
VISUALIZED_FOLDER = 'visualized'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(VISUALIZED_FOLDER):
    os.makedirs(VISUALIZED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZED_FOLDER'] = VISUALIZED_FOLDER


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

@app.route('/hyperspectral/<filename>', methods=['GET'])
def get_hyperspectral_image(filename):
    try:
        logging.debug(f"Request received for hyperspectral image: {filename}")

        # Extract base filename and ensure corresponding .hdr file exists
        base_filename, ext = os.path.splitext(filename)
        hdr_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.hdr")
        img_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.img")
        visualized_filepath = os.path.join(app.config['VISUALIZED_FOLDER'], f"{base_filename}.png")

        logging.debug(f"Looking for .hdr file at: {hdr_file}")
        logging.debug(f"Looking for .img file at: {img_file}")

        if not (os.path.exists(hdr_file) and os.path.exists(img_file)):
            logging.error(f"File not found: {hdr_file} or {img_file}")
            return jsonify({"error": "File not found"}), 404

        # Check if the visualized image already exists
        if os.path.exists(visualized_filepath):
            logging.debug(f"Visualized image already exists for: {filename}")
        else:
            # Load the hyperspectral image
            hyperspectral_image = open_image(hdr_file)
            hyperspectral_data = hyperspectral_image.load()

            # Convert hyperspectral data to an RGB image for visualization
            rgb_image = hyperspectral_data[:, :, :3]  # Assuming first 3 bands are R, G, B

            # Normalize the image data to 0-255
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255
            rgb_image = rgb_image.astype(np.uint8)

            # Convert the numpy array to an image
            pil_img = Image.fromarray(rgb_image)

            # Save the image to a file
            pil_img.save(visualized_filepath)
            logging.debug(f"Successfully created and saved RGB image for: {filename}")

        # Send the saved image file to the client
        return send_file(visualized_filepath, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error processing hyperspectral image: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)