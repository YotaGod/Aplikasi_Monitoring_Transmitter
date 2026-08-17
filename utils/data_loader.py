import pandas as pd
import joblib
import streamlit as st
import os
import numpy as np

@st.cache_data
def load_clean_data():
    """Load data ASLI untuk ditampilkan ke operator"""
    df = pd.read_csv('data/transmitter_data_cleaned.csv')
    return df

@st.cache_data
def load_preprocessed_data():
    """Load data PREPROCESSED untuk deteksi anomali"""
    df = pd.read_csv('data/transmitter_data_preprocessed.csv')
    return df

@st.cache_resource
def load_model():
    """Load model dan threshold"""
    try:
        model = joblib.load('data/model.pkl')
        try:
            with open('data/threshold.txt', 'r') as f:
                threshold = float(f.read().strip())
        except:
            threshold = -0.2
        return model, threshold
    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {e}")
        st.stop()

def get_combined_data(df_clean, model):
    """Gabungkan data utama dengan data input dan beri prediksi"""
    
    # Mulai dengan data utama
    df_combined = df_clean.copy()
    
    input_file = 'data/transmitter_data_input.csv'
    if os.path.exists(input_file):
        try:
            df_input = pd.read_csv(input_file)
            if len(df_input) > 0:
                # Standarisasi format jam
                df_input['Jam'] = df_input['Jam'].apply(
                    lambda x: x if len(str(x).split(':')) == 3 else str(x) + ':00'
                )
                
                # PREDIKSI DATA INPUT MENGGUNAKAN MODEL
                feature_names = ['Vision Output Power (KW)', 'Beam Voltage (KV)', 
                               'Beam Current (A)', 'Driver FWD Power (W)', 
                               'Water Temp In (C)', 'Water Temp Out (C)']
                
                # Ambil fitur dari data input
                X_input = df_input[feature_names].values
                
                # Prediksi
                input_preds = model.predict(X_input)
                input_scores = model.decision_function(X_input)
                
                # Tambahkan hasil prediksi ke df_input
                df_input['prediction'] = input_preds
                df_input['anomaly_score'] = input_scores
                df_input['anomaly_status'] = df_input['prediction'].map({1: 'Normal', -1: 'Anomali'})
                
                # Gabungkan
                df_combined = pd.concat([df_combined, df_input], ignore_index=True)
                
                # Urutkan berdasarkan tanggal dan jam
                df_combined['datetime'] = pd.to_datetime(
                    df_combined['Tanggal'].astype(str) + ' ' + df_combined['Jam'].astype(str),
                    errors='coerce'
                )
                df_combined = df_combined.dropna(subset=['datetime'])
                df_combined = df_combined.sort_values('datetime').reset_index(drop=True)
                df_combined = df_combined.drop(columns=['datetime'])
                
        except Exception as e:
            st.warning(f"⚠️ Gagal memproses data input: {e}")
    
    return df_combined