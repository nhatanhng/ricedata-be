#             img_path = hsi_to_rgb(img_name, 55, 28, 12)
import logging
import traceback
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models import db, Files, Points, VisualizedImages, RecommendChannels, StatisticalData
import pandas as pd

from PIL import Image
import spectral as sp

from npy_append_array import NpyAppendArray
import numpy as np

from pyproj import Proj, transform
import math
import utm

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'nhatanhng'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ricedata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ALLOWED_EXTENSIONS'] = {'csv'}


db.init_app(app)

logging.basicConfig(level=logging.DEBUG)

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'uploads'
VISUALIZED_FOLDER = 'visualized'
UPLOAD_FOLDER_NPY = 'uploads/npy'
UPLOAD_CSV_FOLDER = 'uploads/csv_mapping_points'


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(VISUALIZED_FOLDER):
    os.makedirs(VISUALIZED_FOLDER)
if not os.path.exists(UPLOAD_FOLDER_NPY):
    os.makedirs(UPLOAD_FOLDER_NPY)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZED_FOLDER'] = VISUALIZED_FOLDER
app.config['UPLOAD_FOLDER_NPY'] = UPLOAD_FOLDER_NPY
app.config['UPLOAD_CSV_FOLDER'] = UPLOAD_CSV_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

        blue = round(np.max(average[0:15]))
        green = round(np.max(average[16:40]))
        red = round(np.max(average[41:85]))
        # nf = np.max(average[86:121])

        file_record = Files.query.filter_by(filename=img.filename).first()
        if not file_record:
            raise ValueError("File not found in the database.")

        recommend_channel = RecommendChannels.query.filter_by(file_id=file_record.id).first()

        if recommend_channel:
            recommend_channel.R = red
            recommend_channel.G = green
            recommend_channel.B = blue
            db.session.commit()
        else:
            recommend_channel = RecommendChannels(
                file_id=file_record.id,
                R=red,
                G=green,
                B=blue,
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

# def convert_northing_easting_to_lat_long(easting, northing, zone_number=48, northern_hemisphere=True):
#     # Define the projection for UTM zone 48N
#     proj_utm = Proj(proj='utm', zone=zone_number, ellps='WGS84', south=not northern_hemisphere)
    
#     # Define the projection for latitude and longitude
#     proj_lat_long = Proj(proj='latlong', datum='WGS84')
    
#     # Convert easting and northing to latitude and longitude
#     longitude, latitude = transform(proj_utm, proj_lat_long, easting, northing)
    
#     return latitude, longitude

# # def latlon_to_pixels(latitude, longitude, map_width, map_height):
#     # Convert longitude to x
#     x = (longitude + 180) / 360 * map_width
    
#     # Convert latitude to y using the Mercator projection formula
#     lat_rad = math.radians(latitude)
#     merc_n = math.log(math.tan((math.pi / 4) + (lat_rad / 2)))
#     y = (map_height / 2) - (map_width * merc_n / (2 * math.pi))
    
#     return int(x), int(y)

# def calculate_pixel_coordinates(top_left_northing, top_left_easting, point_northing, point_easting, meters_per_pixel):
#     # Calculate the difference in UTM coordinates
#     delta_easting = point_easting - top_left_easting
#     delta_northing = top_left_northing - point_northing  # Subtract because moving down
    
#     # Convert the UTM differences to pixel offsets
#     pixel_x = int(delta_easting / meters_per_pixel)
#     pixel_y = int(delta_northing / meters_per_pixel)
    
#     return pixel_x, pixel_y

# def adjust_coordinates_to_fit(image_width, image_height, all_pixel_coords):
#     min_x, min_y = float('inf'), float('inf')
    
#     # Find the minimum pixel coordinates to determine the offset needed
#     for pixel_x, pixel_y in all_pixel_coords:
#         min_x = min(min_x, pixel_x)
#         min_y = min(min_y, pixel_y)
    
#     # Calculate the necessary offset to bring all coordinates within the image bounds
#     x_offset = -min_x if min_x < 0 else 0
#     y_offset = -min_y if min_y < 0 else 0
    
#     return x_offset, y_offset


# def calculate_and_store_pixel_coordinates():
#     base_point = StatisticalData.query.filter_by(point_id='BASE').first()
    
#     if not base_point:
#         print("BASE point not found.")
#         return

#     top_left_northing = base_point.y  # Assuming y represents northing in the database
#     top_left_easting = base_point.x  # Assuming x represents easting in the database
    
#     # Assume meters per pixel is known
#     meters_per_pixel = 0.03  # Example resolution, adjust based on your data

#     all_points = StatisticalData.query.filter(StatisticalData.point_id != 'BASE').all()
    
#     all_pixel_coords = []

#     for point in all_points:
#         visualized_image = VisualizedImages.query.filter_by(id=point.image_id).first()
#         image_width = visualized_image.width
#         image_height = visualized_image.height

#         # # Convert northing and easting to latitude and longitude
#         # latitude, longitude = convert_northing_easting_to_lat_long(point.x, point.y)
#         # print(point.x, point.y)

#         # # Convert latitude and longitude to pixel coordinates
#         # pixel_x, pixel_y = latlon_to_pixels(latitude, longitude, image_width, image_height)
#         # print(pixel_x, pixel_y)

#         # Convert point northing/easting to pixel coordinates
#         pixel_x, pixel_y = calculate_pixel_coordinates(top_left_northing, top_left_easting, point.y, point.x, meters_per_pixel)
#         all_pixel_coords.append((pixel_x, pixel_y))

#     # Adjust all coordinates to fit within the image dimensions
#     x_offset, y_offset = adjust_coordinates_to_fit(image_width, image_height, all_pixel_coords)
   


#     #     new_point = Points(
#     #         image_id=point.image_id,
#     #         point_id=point.point_id,
#     #         x=pixel_x,
#     #         y=pixel_y
#     #     )
        
#     #     db.session.add(new_point)
    
#     # db.session.commit()

#     # Store the adjusted coordinates in the database
#     for i, point in enumerate(all_points):
#         pixel_x, pixel_y = all_pixel_coords[i]
        
#         # Apply the offset
#         pixel_x += x_offset
#         pixel_y += y_offset
        
#         new_point = Points(
#             image_id=point.image_id,
#             point_id=point.point_id,
#             x=pixel_x,
#             y=pixel_y
#         )
        
#         db.session.add(new_point)
    
#     db.session.commit()
#     print("Pixel coordinates calculated, converted to pixel value, and stored successfully.")

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
        

        visualized_images = VisualizedImages.query.filter_by(file_id=file.id).all()
        for visualized_image in visualized_images:
            points = Points.query.filter_by(image_id=visualized_image.id).all()
            for point in points:
                db.session.delete(point)
            
            statistical_data = StatisticalData.query.filter_by(image_id=visualized_image.id).all()
            for data in statistical_data:
                db.session.delete(data)
            
            db.session.delete(visualized_image)
        
        recommend_channels = RecommendChannels.query.filter_by(file_id=file.id).all()
        for recommend_channel in recommend_channels:
            db.session.delete(recommend_channel)
        
        db.session.delete(file)
        db.session.commit()

        if os.path.exists(file.filepath):
            os.remove(file.filepath)
        else:
            return jsonify({"message": f"File {filename} deleted from database, but file was not found on disk"}), 200
        
        return jsonify({"message": f"File {filename} deleted successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error deleting file {filename}: {str(e)}")
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
        
        img_name = filename.split('.')[0]
        img_path = os.path.join(VISUALIZED_FOLDER, img_name + '.png')
        
        img_path = hsi_to_rgb(img_name, r, g, b)
        logging.info(f"Image {img_name}.png created and saved with R={r}, G={g}, B={b}.")

        # Open the image to get its dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        
        file_record = Files.query.filter_by(filename=filename).first()
        if file_record:
            visualized_image = VisualizedImages.query.filter_by(file_id=file_record.id).first()
            if visualized_image:
                visualized_image.visualized_filepath = img_path
                visualized_image.width = width  
                visualized_image.height = height  
            else:
                visualized_image = VisualizedImages(
                    file_id=file_record.id,
                    visualized_filename=img_name + '.png',
                    visualized_filepath=img_path,
                    width=width,  
                    height=height  
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

    recommend_channel = RecommendChannels.query.filter_by(file_id=file_record.id).first()
    if not recommend_channel:
        print(f"Recommendation channel not found for filename: {file_record.filename}")
        return jsonify({"error": "Recommendation channel not found"}), 404

    print(f"Recommendation channel found: R={recommend_channel.R}, G={recommend_channel.G}, B={recommend_channel.B}")

    return jsonify({
        "R": recommend_channel.R,
        "G": recommend_channel.G,
        "B": recommend_channel.B
    })  


@app.route('/visualized_files', methods=['GET'])
def get_visualized_files():
    try:
        visualized_images = VisualizedImages.query.all()
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
        if not filename.endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'

        print(filename)
        visualized_image = VisualizedImages.query.filter_by(visualized_filename=filename).first()
        if not visualized_image:
            return jsonify({"error": "File record not found"}), 404

        points = request.json.get('points', [])
        
        Points.query.filter_by(image_id=visualized_image.id).delete()

        for point in points:
            new_point = Points(
                image_id=visualized_image.id,
                x=point['x'],
                y=point['y']
            )
            db.session.add(new_point)

        db.session.commit()
        return jsonify({"message": "Points saved successfully"}), 200

    except Exception as e:
        logging.error(f"Error saving points: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/delete_point/<int:point_id>', methods=['DELETE'])
def delete_point(point_id):
    try:
        point = Points.query.get(point_id)
        if not point:
            return jsonify({"error": "Point not found"}), 404

        db.session.delete(point)
        db.session.commit()
        return jsonify({"message": "Point deleted successfully"}), 200
    except Exception as e:
        logging.error(f"Error deleting point: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files or 'image_id' not in request.form:
        return jsonify({"error": "No file or image_id provided"}), 400
    
    file = request.files['file']
    image_filename = request.form['image_id'] 
    print(image_filename) 

    if image_filename.endswith('.img'):
        image_filename = image_filename.replace('.img', '.png')

    visualized_image = VisualizedImages.query.filter_by(visualized_filename=image_filename).first()

    if not visualized_image:
        return jsonify({"error": "Visualized image not found"}), 404
    
    image_id = visualized_image.id  

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_CSV_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"{file.filename} uploaded succesfully")
        
        try:
            data = pd.read_csv(file_path,delimiter=';')

            # Convert date strings to Python date objects
            data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y').dt.date

            # Assuming the CSV has columns matching the fields in StatisticalData
            for index, row in data.iterrows():
                point_id = row['ID']
            
                new_entry = StatisticalData(
                    image_id=image_id,
                    point_id=point_id,
                    y=row['X(m)'],
                    x =row['Y(m)'],
                    h=row.get('H(m)_EGM96'),
                    replicate=row.get('replicate'),
                    sub_replicate=row.get('sub_replicate'),
                    chlorophyll=row.get('chlorophyll'),
                    rice_height=row.get('rice_height'),
                    spectral_num=row.get('spectral_num'),
                    digesion=row.get('digesion'),
                    p_conc=row.get('P_conc'),
                    k_conc=row.get('K_conc'),
                    n_conc=row.get('N_conc'),
                    chlorophyll_a=row.get('Chlorophyll_a'),
                    date=row.get('date')
                )
                db.session.add(new_entry)

            db.session.commit()
            # logging.info("CSV data uploaded and added successfully.")

            # calculate_and_store_pixel_coordinates()
            # logging.info("pixel coordinated calculated and stored.")

            return jsonify({"message": "CSV data uploaded and added successfully."}), 200

        except Exception as e:
            logging.error(f"Error processing CSV: {e}")
            logging.error(traceback.format_exc())
            return jsonify({"error": "An error occurred while processing the CSV file."}), 500

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/get_point_ids', methods=['GET'])
def get_point_ids():
    try:
        # Retrieve all unique point IDs from the StatisticalData table
        point_ids = db.session.query(StatisticalData.point_id).distinct().all()

        # Flatten the list of tuples
        point_ids = [point_id[0] for point_id in point_ids]

        return jsonify({"point_ids": point_ids}), 200

    except Exception as e:
        logging.error(f"Error retrieving point IDs: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/store_point_id', methods=['POST'])
def store_point_id():
    try:
        # Get the selected point details from the request
        selected_point = request.json.get('selected_point', {})
        print(f"Received selected_point: {selected_point}")
        
        point_id = selected_point.get('point_id')
        x = selected_point.get('x')
        y = selected_point.get('y')
        image_id = selected_point.get('image_id')

        print(f"Extracted point_id: {point_id}, x: {x}, y: {y}, image_id: {image_id}")

        if not point_id:
            return jsonify({"error": "No point_id provided"}), 400

        # Resolve the image_id from the image filename
        visualized_image = VisualizedImages.query.filter_by(visualized_filename=image_id.replace('.img', '.png')).first()

        if not visualized_image:
            print(f"No VisualizedImages entry found for visualized_filename: {image_id}")
            return jsonify({"error": "Invalid image_id provided"}), 400
        
        image_id_num = visualized_image.id
        print(f"Resolved image_id_num: {image_id_num}")

        # Find the corresponding point in the Points table
        existing_point = Points.query.filter_by(
            image_id=image_id_num,
            x=x,
            y=y
        ).first()

        if existing_point:
            # Update the point_id if necessary
            existing_point.point_id = point_id
            print(f"Updated existing Point: {existing_point}")
        else:
            # Insert a new entry into the Points table
            new_point = Points(
                image_id=image_id_num,
                point_id=point_id,
                x=x,
                y=y
            )
            db.session.add(new_point)
            print(f"Added new Point: {new_point}")

        # Commit the changes to the database
        db.session.commit()

        return jsonify({"message": "Point ID stored successfully"}), 200

    except Exception as e:
        logging.error(f"Error storing point ID: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_statistical_data', methods=['POST'])
def get_statistical_data():
    try:
        # Get the selected point_id from the request
        point_id = request.json.get('point_id')
        print(f"Received point_id: {point_id}")

        if not point_id:
            return jsonify({"error": "No point_id provided"}), 400

        # Retrieve the StatisticalData entry that matches the provided point_id
        statistical_data_entry = StatisticalData.query.filter_by(point_id=point_id).first()

        if not statistical_data_entry:
            print(f"No StatisticalData entry found for point_id: {point_id}")
            return jsonify({"error": "No data found for the given point_id"}), 404

        # Convert the data to a dictionary to return as JSON
        data = {
            "id": statistical_data_entry.id,
            "image_id": statistical_data_entry.image_id,
            "point_id": statistical_data_entry.point_id,
            "x": statistical_data_entry.x,
            "y": statistical_data_entry.y,
            "h": statistical_data_entry.h,
            "replicate": statistical_data_entry.replicate,
            "sub_replicate": statistical_data_entry.sub_replicate,
            "chlorophyll": statistical_data_entry.chlorophyll,
            "rice_height": statistical_data_entry.rice_height,
            "spectral_num": statistical_data_entry.spectral_num,
            "digesion": statistical_data_entry.digesion,
            "p_conc": statistical_data_entry.p_conc,
            "k_conc": statistical_data_entry.k_conc,
            "n_conc": statistical_data_entry.n_conc,
            "chlorophyll_a": statistical_data_entry.chlorophyll_a,
            "date": statistical_data_entry.date
        }

        print(f"Returning statistical data: {data}")
        return jsonify({"statistical_data": data}), 200

    except Exception as e:
        logging.error(f"Error retrieving statistical data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_points/<image_filename>', methods=['GET'])
def get_points(image_filename):
    try:
        visualized_image = VisualizedImages.query.filter_by(visualized_filename=image_filename).first()

        if not visualized_image:
            return jsonify({"error": "Image not found"}), 404

        points = Points.query.filter_by(image_id=visualized_image.id).all()

        points_data = [
            {
                "id": point.id,
                "x": point.x,
                "y": point.y,
                "point_id": point.point_id
            }
            for point in points
        ]

        return jsonify(points_data), 200

    except Exception as e:
        logging.error(f"Error retrieving points: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
