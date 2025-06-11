from flask import Blueprint, request, jsonify, current_app
from keras.models import load_model
import joblib
import numpy as np
from datetime import datetime

prediksi_bp = Blueprint('prediksi_bp', __name__)

# Pemuatan Model
try:
    model_pemasukan = load_model('model.h5')
    model_pengeluaran = load_model('model2.h5')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    scaler_x_pengeluaran = joblib.load('scaler_x_pengeluaran.pkl')
    scaler_y_pengeluaran = joblib.load('scaler_y_pengeluaran.pkl')
    print("Models dan scalers untuk prediksi berhasil dimuat.")
except Exception as e:
    print(f"Peringatan: Error saat memuat model/scaler, endpoint prediksi tidak akan berfungsi. Error: {e}")
    model_pemasukan, model_pengeluaran = None, None

# Fungsi Helper Internal untuk Menghindari Duplikasi Kode
def _run_prediction(data, model, scaler_x_instance, scaler_y_instance):
    """
    Fungsi generik untuk menjalankan validasi, preprocessing, dan prediksi.
    """
    window_size = model.input_shape[1] 
    forecast_size = model.output_shape[1]
    
    # Menggunakan key 'data_terakhir' yang generik
    if not data or 'data_terakhir' not in data:
        return {'error': 'Input JSON harus berisi key "data_terakhir"'}, 400

    last_data_points = data['data_terakhir']
    if not isinstance(last_data_points, list) or len(last_data_points) != window_size:
        return {'error': f'Key "data_terakhir" harus berisi list dengan {window_size} angka'}, 400

    # Preprocessing
    features = np.array(last_data_points).reshape(-1, 1)
    features_scaled = scaler_x_instance.transform(features)
    features_scaled_3d = features_scaled.reshape(1, window_size, 1)

    # Prediksi
    pred_scaled = model.predict(features_scaled_3d)
    pred_original = scaler_y_instance.inverse_transform(pred_scaled)
    
    # Format hasil
    prediksi_berikutnya = pred_original.flatten().tolist()
    
    return {
        f'prediksi_{forecast_size}_hari_kedepan': prediksi_berikutnya,
        'timestamp': datetime.now().isoformat()
    }, 200

# Endpoint Prediksi
@prediksi_bp.route('/predict/pemasukan', methods=['POST'])
def predict_pemasukan():
    if model_pemasukan is None:
        return jsonify({'error': 'Model Pemasukan tidak tersedia di server.'}), 503
    try:
        result, status_code = _run_prediction(
            request.get_json(), model_pemasukan, scaler_x, scaler_y
        )
        return jsonify(result), status_code
    except Exception as e:
        current_app.logger.error(f"Error di /predict/pemasukan: {e}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan internal saat prediksi pemasukan.'}), 500

@prediksi_bp.route('/predict/pengeluaran', methods=['POST'])
def predict_pengeluaran():
    if model_pengeluaran is None:
        return jsonify({'error': 'Model Pengeluaran tidak tersedia di server.'}), 503
    try:
        result, status_code = _run_prediction(
            request.get_json(), model_pengeluaran, scaler_x_pengeluaran, scaler_y_pengeluaran
        )
        return jsonify(result), status_code
    except Exception as e:
        current_app.logger.error(f"Error di /predict/pengeluaran: {e}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan internal saat prediksi pengeluaran.'}), 500