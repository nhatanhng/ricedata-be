from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Files(db.Model):
    __tablename__ = "files"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(200), nullable=False) 
    points = db.relationship('Point', backref='file', lazy=True)
    visualized_images = db.relationship('VisualizedImage', backref='files', lazy=True)

class Point(db.Model):
    __tablename__ = "points"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    file_id = db.Column(db.Integer, db.ForeignKey('files.id'), nullable=False) 
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)

class VisualizedImage(db.Model):
    __tablename__ = "visualized_images"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    file_id = db.Column(db.Integer, db.ForeignKey('files.id'), nullable=False)
    visualized_filename = db.Column(db.String(255), nullable=False)
    visualized_filepath = db.Column(db.String(200), nullable=False)
