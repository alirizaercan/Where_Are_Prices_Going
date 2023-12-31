import pickle

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('binden_yuksekler.csv')


print(data.shape)

data['tarih'] = pd.to_datetime(data['tarih'])
data.set_index('tarih', inplace=True)

def create_windowed_dataset(data, window_size=3):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


df = data[data['urun_ad'] == 'ARMUT( Ankara )'][['ortalama_fiyat']]
print(df)
cut_off_date = df.index.max() - pd.DateOffset(years=4)
train_df = df[df.index <= cut_off_date]
test_df = df[df.index > cut_off_date]

print('train_df shape:', train_df.shape)
print(df.index.min())
print(df.index.max())

if not train_df.empty:
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    window_size = 3
    X_train, y_train = create_windowed_dataset(train_scaled, window_size)
    X_test, y_test = create_windowed_dataset(test_scaled, window_size)

    # LSTM
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=100)

    train_predictions = model.predict(X_train).flatten()
    test_predictions = model.predict(X_test).flatten()

    pickle_file_path = 'model-windows-3-years-6-500.pkl'  # Replace with your desired file path
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(model, file)

    plt.figure(figsize=(15, 6))

    plt.plot(df.index, df['ortalama_fiyat'], label='Gerçek Değerler', color='blue')

    train_index = train_df.index[window_size:len(train_predictions)+window_size]
    plt.plot(train_index, train_predictions, label='Eğitim Tahminleri', color='green')

    test_index = test_df.index[window_size:len(test_predictions)+window_size]
    plt.plot(test_index, test_predictions, label='Test Tahminleri', color='red')

    plt.title('Model Tahminleri ve Gerçek Değerler')
    plt.xlabel('Tarih')
    plt.ylabel('Ortalama Fiyat')
    plt.legend()
    plt.show()

else: 
    print("No data in train_df. Check your filtering conditions.")
