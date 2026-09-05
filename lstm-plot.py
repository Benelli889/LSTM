import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


# 1. Beispieldaten erzeugen (ersetzen Sie dies durch Ihre echten Beschleunigungsdaten)
# z.B. 1000 Messpunkte
# Frequenz um Faktor 2 reduzieren

def generate_raw_signal(time_steps=1000):

  time_steps = 1000
  raw_signal = np.sin(np.linspace(0, 10, time_steps)) + np.random.normal(
      0, .1, time_steps
  )
  raw_signal = raw_signal.reshape(-1, 1)

  # Graph speichern: Rohsignal
  plt.figure(figsize=(12, 4))
  plt.plot(raw_signal, linewidth=1, color='blue')
  plt.title('Rohsignal (raw_signal)')
  plt.xlabel('Zeit (Schritte)')
  plt.ylabel('Amplitude')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('00_raw_signal.png', dpi=150, bbox_inches='tight')
  print("Graph gespeichert als '00_raw_signal.png'")
  plt.close()

  # 2. Skalierung
  #scaler = MinMaxScaler(feature_range=(-1, 1))
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_signal = scaler.fit_transform(raw_signal)

  # Graph speichern: Skaliertes Signal
  plt.figure(figsize=(12, 4))
  plt.plot(scaled_signal, linewidth=1)
  plt.title('Skaliertes Signal')
  plt.xlabel('Zeit (Schritte)')
  plt.ylabel('Amplitude (skaliert)')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('01_scaled_signal.png', dpi=150, bbox_inches='tight')
  print("Graph gespeichert als '01_scaled_signal.png'")
  plt.close()

  # Graph speichern: Vergleich Original vs. Skaliert
  fig, axes = plt.subplots(2, 1, figsize=(12, 6))

  axes[0].plot(raw_signal, linewidth=1, color='blue')
  axes[0].set_title('Original Signal')
  axes[0].set_ylabel('Amplitude')
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(scaled_signal, linewidth=1, color='orange')
  axes[1].set_title('Skaliertes Signal (MinMax: -1 bis 1)')
  axes[1].set_ylabel('Amplitude (skaliert)')
  axes[1].set_xlabel('Zeit (Schritte)')
  axes[1].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('02_original_vs_scaled.png', dpi=150, bbox_inches='tight')
  print("Graph gespeichert als '02_original_vs_scaled.png'")
  plt.close()

  # 3. Sliding-Window-Datensatz erstellen (Lookback-Fenster)
  def create_dataset(data, window_size=50):
    X, y = [], []
    for i in range(len(data) - window_size):
      X.append(data[i : i + window_size])
      y.append(data[i + window_size])
    return np.array(X), np.array(y)


  WINDOW_SIZE = 200  # Letzte 50 Zeitschritte betrachten
  X, y = create_dataset(scaled_signal, window_size=WINDOW_SIZE)

  # Train/Test Split (z. B. 80% Training, 20% Test)
  split = int(len(X) * 0.9)
  X_train, X_test = X[:split], X[split:]
  y_train, y_test = y[:split], y[split:]

  # 4. Modell aufbauen (hier mit LSTM, für GRU einfach LSTM durch GRU ersetzen)
  model = Sequential([
      #LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
      LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
      Dropout(0.2),
      #LSTM(32),
      LSTM(64),
      Dense(1),  # Vorhersage des nächsten Beschleunigungswerts
  ])
  # model = Sequential([
  #     GRU(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
  #     Dropout(0.2),
  #     GRU(32),
  #     Dense(1),  # Vorhersage des nächsten Beschleunigungswerts
  # ])

  model.compile(optimizer='adam', loss='mean_squared_error')
  model.summary()

  # 5. Training
  history = model.fit(
      X_train,
      y_train,
      epochs=20,
      batch_size=32,
      validation_data=(X_test, y_test),
      verbose=1,
  )

  # 6. Vorhersage & Rückskalierung
  predictions_scaled = model.predict(X_test)
  predictions = scaler.inverse_transform(predictions_scaled)

  print(predictions_scaled)
  print(predictions)

  # 7. Visualisierung: predictions_scaled und predictions
  fig, axes = plt.subplots(2, 1, figsize=(14, 8))

  # Oben: Skalierte Vorhersagen
  axes[0].plot(y_test, label='Echte Werte (skaliert)', linewidth=2, color='blue', alpha=0.7)
  axes[0].plot(predictions_scaled, label='Vorhersagen (skaliert)', linewidth=2, color='orange', alpha=0.7)
  axes[0].set_title('Skalierte Vorhersagen vs. Echte Werte (Range: -1 bis 1)', fontsize=12, fontweight='bold')
  axes[0].set_ylabel('Amplitude (skaliert)')
  axes[0].legend(loc='best')
  axes[0].grid(True, alpha=0.3)

  # Unten: Rückskalierte Vorhersagen
  y_test_original = scaler.inverse_transform(y_test)
  axes[1].plot(y_test_original, label='Echte Werte (Original)', linewidth=2, color='blue', alpha=0.7)
  axes[1].plot(predictions, label='Vorhersagen (Original)', linewidth=2, color='green', alpha=0.7)
  axes[1].set_title('Rückskalierte Vorhersagen vs. Echte Werte (Original Range)', fontsize=12, fontweight='bold')
  axes[1].set_xlabel('Test-Probe')
  axes[1].set_ylabel('Amplitude')
  axes[1].legend(loc='best')
  axes[1].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('03_predictions_comparison.png', dpi=150, bbox_inches='tight')
  print("Graph gespeichert als '03_predictions_comparison.png'")
  plt.close()

  # Zusätzliche Metrik: MSE
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_test_original, predictions)
  print(f"Mean Squared Error (Original Scale): {mse:.6f}")

  # 8. Visualisierung: raw_signal mit predictions_scaled
  # Zoome in auf den Test-Bereich für bessere Sichtbarkeit
  fig, axes = plt.subplots(2, 1, figsize=(14, 10))

  # PLOT 1: Gesamtes Signal mit Hervorhebung des Test-Bereichs
  ax1 = axes[0]
  ax1.plot(scaled_signal, label='Gesamtes Signal (scaled_signal)', linewidth=1.5, color='blue', alpha=0.6)
  # Markiere Trainings- und Test-Bereich
  ax1.axvspan(0, split, alpha=0.2, color='green', label='Trainings-Bereich')
  ax1.axvspan(split, len(scaled_signal), alpha=0.2, color='red', label='Test-Bereich')
  ax1.set_title('Gesamtes Signal mit Train/Test Bereich', fontsize=12, fontweight='bold')
  ax1.set_ylabel('Amplitude (skaliert)')
  ax1.legend(loc='best')
  ax1.grid(True, alpha=0.3)

  # PLOT 2: Zoom in auf Test-Bereich mit Vorhersagen
  ax2 = axes[1]
  # Zeige nur den Test-Bereich des echten Signals
  test_start = split + WINDOW_SIZE
  test_signal_section = scaled_signal[test_start:test_start + len(predictions_scaled)]

  # Zeichne das echte Signal im Test-Bereich
  x_indices = range(len(predictions_scaled))
  ax2.plot(x_indices, test_signal_section.flatten(), 
          label='Echte Werte (Test-Bereich)', 
          linewidth=2.5, 
          color='blue', 
          alpha=0.8,
          marker='s',
          markersize=4)

  # Zeichne die Vorhersagen
  ax2.plot(x_indices, predictions_scaled.flatten(), 
          label='Vorhersagen (predictions_scaled)', 
          linewidth=2.5, 
          color='red', 
          alpha=0.8,
          marker='o',
          markersize=4,
          linestyle='--')

  ax2.set_title('Zoom: Test-Bereich - Echte Werte vs. Vorhersagen', fontsize=12, fontweight='bold')
  ax2.set_xlabel('Proben im Test-Set')
  ax2.set_ylabel('Amplitude (skaliert)')
  ax2.legend(loc='best', fontsize=10)
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('04_signal_with_predictions.png', dpi=150, bbox_inches='tight')
  print("Graph gespeichert als '04_signal_with_predictions.png'")
  plt.close()

  print("\n✅ Alle Graphen wurden als PNG-Dateien gespeichert!")

generate_raw_signal(time_steps=1000)