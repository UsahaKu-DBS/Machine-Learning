import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# Muat environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# Inisialisasi db
db = SQLAlchemy()

def create_app():
    """Application Factory Function"""
    app = Flask(__name__)

    # Mengambil konfigurasi database dari environment variables
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')
    
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Inisialisasi Ekstensi
    db.init_app(app)

    # Impor dan Registrasi Blueprints
    from routes.transaction import transaksi_bp
    from routes.predict import prediksi_bp

    app.register_blueprint(transaksi_bp)
    app.register_blueprint(prediksi_bp)

    # Konfigurasi Logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        # Mencatat log ke dalam file, akan membuat file baru jika sudah 10MB
        file_handler = RotatingFileHandler('logs/usahaku.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('Aplikasi Usahaku dimulai')

    return app

# Entry Point untuk Menjalankan Aplikasi
if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
