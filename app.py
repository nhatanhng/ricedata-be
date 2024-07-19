import logging
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models import db, Files, Point, VisualizedImage
from sqlalchemy import text

from PIL import Image
import spectral as sp
import spectral.io.envi as envi

from spectral import open_image
from models import db, Files
from npy_append_array import NpyAppendArray
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'nhatanhng'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ricedata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

with app.app_context():
    db.create_all()

# Directory to save uploaded files and visualized images
UPLOAD_FOLDER = 'uploads'
VISUALIZED_FOLDER = 'visualized'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(VISUALIZED_FOLDER):
    os.makedirs(VISUALIZED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZED_FOLDER'] = VISUALIZED_FOLDER

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
        Point.query.filter_by(filename=filename).delete()  # Clear points associated with the file
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

        # Check if the visualized image already exists in the database
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            logging.error(f"File record not found: {filename}")
            return jsonify({"error": "File record not found"}), 404
        
        visualized_image = VisualizedImage.query.filter_by(file_id=file.id).first()
        if visualized_image:
            logging.debug(f"Visualized image already exists for: {filename}")
            visualized_filepath = visualized_image.visualized_filepath
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

            # Save visualized image information to the database
            new_visualized_image = VisualizedImage(
                file_id=file.id,
                visualized_filename=f"{base_filename}.png",
                visualized_filepath=visualized_filepath
            )
            db.session.add(new_visualized_image)
            db.session.commit()

        # Send the saved image file to the client
        return send_file(visualized_filepath, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error processing hyperspectral image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualized_files', methods=['GET'])
def get_visualized_files():
    try:
        visualized_images = VisualizedImage.query.all()
        visualized_filenames = [img.visualized_filename for img in visualized_images]
        return jsonify(visualized_filenames), 200
    except Exception as e:
        logging.error(f"Error fetching visualized files: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualized/<filename>', methods=['GET'])
def get_visualized_file(filename):
    try:
        file_path = os.path.join(app.config['VISUALIZED_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error serving visualized file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_points/<filename>', methods=['POST'])
def save_points(filename):
    try:
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File record not found"}), 404

        points = request.json.get('points', [])
        
        Point.query.filter_by(file_id=file.id).delete()

        for point in points:
            new_point = Point(
                file_id=file.id,
                x=point['x'],
                y=point['y']
            )
            db.session.add(new_point)

        db.session.commit()
        return jsonify({"message": "Points saved successfully"}), 200

    except Exception as e:
        logging.error(f"Error saving points: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_points/<filename>', methods=['GET'])
def get_points(filename):
    try:
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File record not found"}), 404
        points = Point.query.filter_by(file_id=file.id).all()
        return jsonify([{'x': point.x, 'y': point.y} for point in points]), 200
    except Exception as e:
        logging.error(f"Error fetching points: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/delete_point/<int:point_id>', methods=['DELETE'])
def delete_point(point_id):
    try:
        point = Point.query.get(point_id)
        if not point:
            return jsonify({"error": "Point not found"}), 404

        db.session.delete(point)
        db.session.commit()
        return jsonify({"message": "Point deleted successfully"}), 200
    except Exception as e:
        logging.error(f"Error deleting point: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
