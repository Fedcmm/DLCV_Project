# Code to extract labels
import os
import pandas as pd
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# Funzione per caricare, trasformare e applicare il filtro
def process_data(file_path, fs=700, lowcut=0.5, highcut=45.0, duration=20):
    # Leggi il file di dati
   # data = pd.read_csv(file_path, skiprows=3, delimiter='\t', usecols=range(2, 10))
   # df_resp = pd.DataFrame(data)
   # df_resp.columns = data.columns.str.strip()
   # df_resp.columns = ["ECG", "EDA", "EMG", "TEMP", "XYZ_X", "XYZ_Y", "XYZ_Z", "RESPIRATION"]

    df_resp=file_path
    # Trasformazioni per i vari segnali
    def transform_ecg(signal):
        return (signal / 2**16 - 0.5) * 3

    def transform_eda(signal):
        return ((signal / 2**16) * 3) / 0.12

    def transform_emg(signal):
        return (signal / 2**16 - 0.5) * 3

    def transform_temp(signal):
        vout = (signal * 3) / (2**16 - 1)
        rntc = (10**4 * vout) / (3 - vout)
        return -273.15 + 1 / (1.12764514e-3 + 2.34282709e-4 * np.log(rntc) + 8.77303013e-8 * np.log(rntc)**3)

    def transform_xyz(signal):
        return (signal - 28000) / (38000 - 28000) * 2 - 1

    def transform_respiration(signal):
        return (signal / 2**16 - 0.5) * 100

    # Applicazione delle trasformazioni
    df_resp['ECG'] = transform_ecg(df_resp['ECG'])
    df_resp['EDA'] = transform_eda(df_resp['EDA'])
    df_resp['EMG'] = transform_emg(df_resp['EMG'])
    df_resp['TEMP'] = transform_temp(df_resp['TEMP'])
    df_resp['XYZ_X'] = transform_xyz(df_resp['XYZ_X'])
    df_resp['XYZ_Y'] = transform_xyz(df_resp['XYZ_Y'])
    df_resp['XYZ_Z'] = transform_xyz(df_resp['XYZ_Z'])
    df_resp['RESPIRATION'] = transform_respiration(df_resp['RESPIRATION'])

    # Funzione per il filtro passa banda
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        y = filtfilt(b, a, data)
        return y
    def butter_highpass(data,cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        return filtfilt(b,a,data)
    def butter_lowwpass(data,cutoff,fs,order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b,a,data)
    # Applicazione del filtro
    fs=700
    ecg_cut=0.5
    emg_cut=10
    lowcut=0.5
    highcut=10
    df_resp['ECG']= apply_bandpass_filter(df_resp['ECG'], lowcut, highcut, fs)
    df_resp['EMG']= apply_bandpass_filter(df_resp['EMG'], lowcut, highcut, fs)


    # Normalizzazione dei dati
    scaler = MinMaxScaler(feature_range=(-1, 1))
    columns_to_normalize = ['ECG', 'EDA', 'EMG', 'TEMP', 'XYZ_X', 'XYZ_Y', 'XYZ_Z', 'RESPIRATION']
    df_resp[columns_to_normalize] = scaler.fit_transform(df_resp[columns_to_normalize])

    # Selezione dei dati per i primi 20 secondi
    num_samples = fs * duration
    df_20s = df_resp.iloc[:num_samples]

    return df_resp

def time_to_tick(time_str: str):
    time_comp = list(map(int, time_str.split('.')))
    return time_comp[0] * 60 * 700 + (time_comp[0] * 700 if len(time_comp) > 1 else 0)

def get_labeled_df(subject: str, class_type: Literal['']):
    with open(os.path.join(source_path, subject, f'{subject}_quest.csv'), 'r') as quest:
        lines = quest.readlines()
        start = lines[2].split(';', 9)[1:-4]
        end = lines[3].split(';', 9)[1:-4]
        start = list(map(time_to_tick, start))
        end = list(map(time_to_tick, end))
        # print(start, end)

    data = pd.read_csv(os.path.join(source_path, subject, f'{subject}_respiban.txt'), skiprows=3, delimiter='\t', usecols=range(0, 10))
    data.columns = ["TICK", "ignore", "ECG", "EDA", "EMG", "TEMP", "XYZ_X", "XYZ_Y", "XYZ_Z", "RESPIRATION"]
    data.drop('ignore', axis='columns', inplace=True)

    # print(data.info())

    labeled = data[((data.TICK >= start[0]) & (data.TICK <= end[0])) |
                ((data.TICK >= start[1]) & (data.TICK <= end[1])) |
                ((data.TICK >= start[2]) & (data.TICK <= end[2])) |
                ((data.TICK >= start[3]) & (data.TICK <= end[3])) |
                ((data.TICK >= start[4]) & (data.TICK <= end[4]))]
    # labeled.describe()

    labels = (['base'] * (end[0] - start[0] + 1) +
            ['fun'] * (end[1] - start[1] + 1) +
            ['medi'] * (end[2] - start[2] + 1) +
            ['stress'] * (end[3] - start[3] + 1) +
            ['medi'] * (end[4] - start[4] + 1))
    labeled = labeled.assign(LABEL=labels)
    return labeled
    # labeled.head()
labeled_dataframes = {}
source_path = 'WESAD'

for subject in os.listdir(source_path):
    if not os.path.isdir(os.path.join(source_path, subject)):
        continue
    labeled_dataframes[subject] = get_labeled_df(subject, 'binary')
    labeled_dataframes[subject] = process_data(labeled_dataframes[subject])
    df = labeled_dataframes[subject]

def prepare_data(df, window_size, step_size, label_column="LABEL"):
    """
    Splits the dataset into time windows suitable for RNN input.
    """
    signals = df.drop(columns=[label_column, "TICK"]).values
    labels = df[label_column].values

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    sequences = []
    sequence_labels = []

    for i in range(0, len(signals) - window_size, step_size):
        sequences.append(signals[i:i + window_size])
        sequence_labels.append(encoded_labels[i + window_size - 1])  # Last label in the window

    return np.array(sequences), np.array(sequence_labels), label_encoder

# Define the output directory
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
window_size = 700  # 1 secondo a 700Hz
step_size = 350
for subject in os.listdir(source_path):
    if not os.path.isdir(os.path.join(source_path, subject)):
        continue
    
    labeled_dataframes[subject] = get_labeled_df(subject, 'binary')
    labeled_dataframes[subject] = process_data(labeled_dataframes[subject])
    df = labeled_dataframes[subject]
    X,y,label_encoder=prepare_data(df,window_size,step_size)

    # Save to .npy file
    X_output_path = os.path.join(output_dir, f"{subject}_processed_X.npy")
    y_output_path = os.path.join(output_dir, f"{subject}_processed_y.npy")
    label_encoder_output_path = os.path.join(output_dir, f"{subject}_processed_label.npy")
    np.save(X_output_path, X)
    np.save(y_output_path, y)
    np.save(label_encoder_output_path, label_encoder)
    

