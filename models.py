from flask_sqlalchemy import SQLAlchemy
 
db = SQLAlchemy()
 
# create datatable
class Files(db.Model):
    __tablename__ = "files"
    id = db.Column(db.Integer, primary_key=True, unique=True,  nullable = False)
    filename = db.Column(db.String(255), nullable = False)
    filepath = db.Column(db.String(200), nullable=False)  # Store file paths instead of file data
    