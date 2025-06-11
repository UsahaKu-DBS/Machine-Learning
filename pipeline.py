from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import os
from datetime import datetime, timedelta
import json
import sqlite3
import mysql.connector
from sqlalchemy import create_engine

app = Flask(__name__)

# Konfigurasi database
DATABASE_CONFIG = {
    'type': 'sqlite',  # 'sqlite', 'mysql', 'postgresql'
    'sqlite_path': 'data.db',
    'mysql': {
        'host': 'localhost',
        'user': 'root',
        'password': 'password',
        'database': 'business_db'
    }
}

# Load model dan scalers
try:
    model_pemasukan = load_model('model.h5')
    model_pengeluaran = load_model('model2.h5')
    
    # Load scalers untuk pemasukan dan pengeluaran
    scaler_x = joblib.load('scaler_x.pkl')  # Scaler untuk input features
    scaler_y = joblib.load('scaler_y.pkl')  # Scaler untuk output pemasukan
    scaler_x_pengeluaran = joblib.load('scaler_x_pengeluaran.pkl')  # Scaler input pengeluaran
    scaler_y_pengeluaran = joblib.load('scaler_y_pengeluaran.pkl')  # Scaler output pengeluaran
        
    print("Models dan scalers berhasil dimuat")
    print(f"Model pemasukan input shape: {model_pemasukan.input_shape}")
    print(f"Model pemasukan output shape: {model_pemasukan.output_shape}")
    print(f"Model pengeluaran input shape: {model_pengeluaran.input_shape}")
    print(f"Model pengeluaran output shape: {model_pengeluaran.output_shape}")
    
except Exception as e:
    print(f"Error loading model/scalers: {e}")
    model_pemasukan = None
    model_pengeluaran = None

# Membuat koneksi ke database berdasarkan konfigurasi
def get_database_connection():
    if DATABASE_CONFIG['type'] == 'sqlite':
        return sqlite3.connect(DATABASE_CONFIG['sqlite_path'])
    elif DATABASE_CONFIG['type'] == 'mysql':
        config = DATABASE_CONFIG['mysql']
        return mysql.connector.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )

# Mengambil data dari database SQL
def get_data_from_sql(query, params=None):
    try:
        if DATABASE_CONFIG['type'] == 'sqlite':
            conn = sqlite3.connect(DATABASE_CONFIG['sqlite_path'])
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
        else:
            # Untuk MySQL/PostgreSQL menggunakan SQLAlchemy
            if DATABASE_CONFIG['type'] == 'mysql':
                config = DATABASE_CONFIG['mysql']
                engine = create_engine(f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}")
                df = pd.read_sql_query(query, engine, params=params)
        
        return df
    except Exception as e:
        print(f"Error mengambil data dari SQL: {e}")
        return None

@app.route('/predict/pemasukan', methods=['POST'])
def predict_pemasukan():
    if model is None:
        return jsonify({'error': 'Model tidak tersedia'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'features' in data:
            features = np.array(data['features'])  # list of list
        else:
          # ubah sesuai yang dibutuhkan
            feature_map = {
                'jumlah_produk_terjual': data.get('jumlah_produk_terjual', 0),
                'harga_jual_rata2': data.get('harga_jual_rata2', 0),
                'jumlah_transaksi': data.get('jumlah_transaksi', 0),
                'biaya_promosi': data.get('biaya_promosi', 0),
                'hari_ke': data.get('hari_ke', 1)
            }
            features = np.array([list(feature_map.values()) for _ in range(30)])  # dummy 30 hari
        
        if len(features.shape) != 2:
            return jsonify({'error': 'Input features harus 2 dimensi [time_steps, features]'}), 400

        features_scaled = scaler_x.transform(features)
        features_scaled = features_scaled.reshape(1, features.shape[0], features.shape[1])
        pred_scaled = model.predict(features_scaled)
        pred_original = scaler_y.inverse_transform(pred_scaled)

        # ubah sesuai yang dibutuhkan
        result = {
            'prediksi_pemasukan': float(pred_original[0][0]),
            'prediksi_pemasukan_formatted': f"Rp {pred_original[0][0]:,.0f}",
            'input_sequence_shape': features.shape,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error prediksi pemasukan: {str(e)}'}), 500

@app.route('/predict/pengeluaran', methods=['POST'])
def predict_pengeluaran():
    if model is None:
        return jsonify({'error': 'Model tidak tersedia'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'features' in data:
            features = np.array(data['features']) 
        else:
          # ubah sesuai yang dibutuhkan
            feature_map = {
                'biaya_bahan_baku': data.get('biaya_bahan_baku', 0),
                'biaya_operasional_lain': data.get('biaya_operasional_lain', 0),
                'biaya_promosi': data.get('biaya_promosi', 0),
                'jumlah_karyawan': data.get('jumlah_karyawan', 0),
                'jumlah_produk_terjual': data.get('jumlah_produk_terjual', 0),
                'hari_ke': data.get('hari_ke', 1),
            }
            single_input = list(feature_map.values())
            features = np.array([single_input for _ in range(30)])  # dummy 30 hari data

        if len(features.shape) != 2:
            return jsonify({'error': 'Input features harus 2 dimensi [time_steps, num_features]'}), 400

        features_scaled = scaler_x_pengeluaran.transform(features)
        features_scaled = features_scaled.reshape(1, features.shape[0], features.shape[1]) 
        pred_scaled = model.predict(features_scaled)
        pred_original = scaler_y_pengeluaran.inverse_transform(pred_scaled)

        # ubah sesuai yang dibutuhkan
        result = {
            'prediksi_pengeluaran': float(pred_original[0][0]),
            'prediksi_pengeluaran_formatted': f"Rp {pred_original[0][0]:,.0f}",
            'input_sequence_shape': features.shape,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error prediksi pengeluaran: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# blm selesai, baru ambil dari sql
