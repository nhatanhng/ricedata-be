import logging
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models import db, Files, Point, VisualizedImage, RecommendChannel

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

logging.basicConfig(level=logging.DEBUG)

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'uploads'
VISUALIZED_FOLDER = 'visualized'
UPLOAD_FOLDER_NPY = 'uploads/npy'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(VISUALIZED_FOLDER):
    os.makedirs(VISUALIZED_FOLDER)
if not os.path.exists(UPLOAD_FOLDER_NPY):
    os.makedirs(UPLOAD_FOLDER_NPY)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZED_FOLDER'] = VISUALIZED_FOLDER
app.config['UPLOAD_FOLDER_NPY'] = UPLOAD_FOLDER_NPY


def npy_converter(img):
    # store the image with its name and npy extension and save it in ./uploads/npy
    npy_filename = "./uploads/npy/" + img.filename.split('.')[0] + '.npy'
    hdr_name = "./uploads/" + img.filename.split('.')[0] + '.hdr'
    hdr_img = sp.envi.open(hdr_name)
    average = []
    try:
        with NpyAppendArray(npy_filename) as npy:
            for i in range(122):
                channel = np.expand_dims(hdr_img.read_band(i), 0)
                average.append(np.average(channel))
                npy.append(channel)

        blue = np.max(average[0:15])
        green = np.max(average[16:40])
        red = np.max(average[41:85])
        # nf = np.max(average[86:121])

        file_record = Files.query.filter_by(filename=img.filename).first()
        if not file_record:
            raise ValueError("File not found in the database.")

        recommend_channel = RecommendChannel.query.filter_by(file_id=file_record.id).first()

        if recommend_channel:
            recommend_channel.R = red
            recommend_channel.G = green
            recommend_channel.B = blue
            # recommend_channel.nf = nf
            db.session.commit()
        else:
            recommend_channel = RecommendChannel(
                file_id=file_record.id,
                R=red,
                G=green,
                B=blue,
                # nf=nf
            )
            db.session.add(recommend_channel)
            db.session.commit()
        
        db.session.add(recommend_channel)
        db.session.commit()

        return jsonify({"message": "File processed and data saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def hsi_to_rgb(hsi_img_name, red, green, blue):
    hsi_img = np.load(os.path.join(UPLOAD_FOLDER_NPY, hsi_img_name + '.npy'))

    red_band = hsi_img[red].astype(np.uint8)
    green_band = hsi_img[green].astype(np.uint8)
    blue_band = hsi_img[blue].astype(np.uint8)

    red_normalized = np.where(red_band > 50, 50, red_band)
    green_normalized = np.where(green_band > 50, 50, green_band)
    blue_normalized = np.where(blue_band > 50, 50, blue_band)

    dr_main_image = np.zeros((red_normalized.shape[0], red_normalized.shape[1], 3), dtype=np.uint8)
    dr_main_image[:, :, 0] = red_normalized
    dr_main_image[:, :, 1] = green_normalized
    dr_main_image[:, :, 2] = blue_normalized

    dr_main_image = (255 * (1.0 / dr_main_image.max() * (dr_main_image - dr_main_image.min()))).astype(np.uint8)

    main_image = Image.fromarray(dr_main_image)
    output_path = os.path.join(VISUALIZED_FOLDER, hsi_img_name + ".png")
    main_image = main_image.save(output_path)

    return output_path


@app.route('/uploads/files', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_extension = filename.split('.')[-1]
            file.save(filepath)

            upload = Files(filename=filename, filepath=filepath, extension=file_extension)
            db.session.add(upload)
            db.session.commit()

            file_record = Files.query.filter_by(filename=filename).first()
            if file_record.extension == 'hdr':
                npy_converter(file)

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
        
        points = Point.query.filter_by(file_id=file.id).all()
        for point in points:
            db.session.delete(point)
        
        visualized_images = VisualizedImage.query.filter_by(file_id=file.id).all()
        for visualized_image in visualized_images:
            db.session.delete(visualized_image)
        
        recommend_channels = RecommendChannel.query.filter_by(file_id=file.id).all()
        for recommend_channel in recommend_channels:
            db.session.delete(recommend_channel)
        
        db.session.delete(file)
        db.session.commit()
        os.remove(file.filepath)
        
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
    
@app.route('/hyperspectral', methods=['POST'])
def visualize_HSI():
    try:
        data = request.json
        filename = data['filename']
        r = data['R']
        g = data['G']
        b = data['B'] 
        # nf = data['nf']
        
        img_name = filename.split('.')[0]
        img_path = os.path.join(VISUALIZED_FOLDER, img_name + '.png')
        
        img_path = hsi_to_rgb(img_name, r, g, b)
        logging.info(f"Image {img_name}.png created and saved with R={r}, G={g}, B={b}.")
        
        file_record = Files.query.filter_by(filename=filename).first()
        if file_record:
            visualized_image = VisualizedImage.query.filter_by(file_id=file_record.id).first()
            if visualized_image:
                visualized_image.visualized_filepath = img_path
            else:
                visualized_image = VisualizedImage(
                    file_id=file_record.id,
                    visualized_filename=img_name + '.png',
                    visualized_filepath=img_path
                )
                db.session.add(visualized_image)
            db.session.commit()

        return send_file(img_path, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error visualizing hyperspectral image: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/recommend_channel/<filename>', methods=['GET'])
def get_recommend_channel(filename):
    print(f"Received request for filename: {filename}")

    file_record = Files.query.filter_by(filename=filename).first()
    if not file_record:
        print(f"File record not found for filename: {filename}")
        return jsonify({"error": "File not found"}), 404

    recommend_channel = RecommendChannel.query.filter_by(file_id=file_record.id).first()
    if not recommend_channel:
        print(f"Recommendation channel not found for filename: {file_record.filename}")
        return jsonify({"error": "Recommendation channel not found"}), 404

    print(f"Recommendation channel found: R={recommend_channel.R}, G={recommend_channel.G}, B={recommend_channel.B}")

    return jsonify({
        "R": recommend_channel.R,
        "G": recommend_channel.G,
        "B": recommend_channel.B
    })  

#             img_path = hsi_to_rgb(img_name, 55, 28, 12)

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
