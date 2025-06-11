from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
import pandas as pd
from pipeline import db 

# Membuat Blueprint
transaksi_bp = Blueprint('transaksi_bp', __name__)

@transaksi_bp.route('/transaksi/umkm/<int:id_umkm>', methods=['GET'])
def get_transaksi_by_umkm(id_umkm):
    """
    Endpoint untuk mengambil daftar transaksi berdasarkan ID UMKM.
    Mendukung filter tanggal dan jenis.
    """
    try:
        filters = {
            'tanggal_mulai': request.args.get('tanggal_mulai'),
            'tanggal_selesai': request.args.get('tanggal_selesai'),
            'jenis': request.args.get('jenis')
        }
        
        base_query = """
            SELECT
                t.*,
                kt.nama_kategori,
                kt.jenis,
                u.nama as created_by_name
            FROM transaksi t
            LEFT JOIN kategori_transaksi kt ON t.id_kategori = kt.id_kategori
            LEFT JOIN users u ON t.created_by = u.id_user
            WHERE t.id_umkm = :id_umkm
        """
        
        params = {'id_umkm': id_umkm}
        
        if filters['tanggal_mulai'] and filters['tanggal_selesai']:
            base_query += " AND t.tanggal_transaksi BETWEEN :tanggal_mulai AND :tanggal_selesai"
            params['tanggal_mulai'] = filters['tanggal_mulai']
            params['tanggal_selesai'] = filters['tanggal_selesai']
        
        if filters['jenis']:
            base_query += " AND kt.jenis = :jenis"
            params['jenis'] = filters['jenis']
            
        base_query += " ORDER BY t.tanggal_transaksi DESC"

        result_df = pd.read_sql_query(text(base_query), db.engine, params=params)
        
        # Konversi kolom tanggal ke format string agar aman untuk JSON
        if 'tanggal_transaksi' in result_df.columns:
            result_df['tanggal_transaksi'] = result_df['tanggal_transaksi'].astype(str)
        
        return jsonify(result_df.to_dict(orient='records'))

    except Exception as e:
        current_app.logger.error(f"Error di /transaksi/umkm/{id_umkm}: {e}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan internal saat mengambil data transaksi.'}), 500

@transaksi_bp.route('/laporan/rekap/umkm/<int:id_umkm>', methods=['GET'])
def get_rekap_by_umkm(id_umkm):
    """
    Endpoint untuk mengambil rekapitulasi total pemasukan, pengeluaran,
    dan saldo bersih untuk UMKM tertentu.
    """
    try:
        filters = {
            'tanggal_mulai': request.args.get('tanggal_mulai'),
            'tanggal_selesai': request.args.get('tanggal_selesai')
        }
        
        query = """
            SELECT
                COALESCE(SUM(CASE WHEN kt.jenis = 'pemasukan' THEN t.jumlah ELSE 0 END), 0) as total_pemasukan,
                COALESCE(SUM(CASE WHEN kt.jenis = 'pengeluaran' THEN t.jumlah ELSE 0 END), 0) as total_pengeluaran,
                COALESCE(SUM(CASE WHEN kt.jenis = 'pemasukan' THEN t.jumlah ELSE -t.jumlah END), 0) as saldo_bersih
            FROM transaksi t
            LEFT JOIN kategori_transaksi kt ON t.id_kategori = kt.id_kategori
            WHERE t.id_umkm = :id_umkm
        """
        params = {'id_umkm': id_umkm}

        if filters['tanggal_mulai'] and filters['tanggal_selesai']:
            query += " AND t.tanggal_transaksi BETWEEN :tanggal_mulai AND :tanggal_selesai"
            params['tanggal_mulai'] = filters['tanggal_mulai']
            params['tanggal_selesai'] = filters['tanggal_selesai']

        result_proxy = db.session.execute(text(query), params)
        result = result_proxy.mappings().first()
        
        if not result:
            return jsonify({'total_pemasukan': 0, 'total_pengeluaran': 0, 'saldo_bersih': 0})
        
        rekap = {key: float(value) for key, value in result.items()}
        
        return jsonify(rekap)

    except Exception as e:
        current_app.logger.error(f"Error di /laporan/rekap/umkm/{id_umkm}: {e}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan internal saat membuat laporan.'}), 500