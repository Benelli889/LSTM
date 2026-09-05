import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# Konfiguration deiner Input-Daten
TIME_STEPS = 60  # Das Netzwerk analysiert immer Fenster von 60 Kerzen
FEATURES = 5     # O, H, L, C, V (oder zusätzliche Pine-Script Indikatoren)

def create_cnnlstm_model():
    model = Sequential([
        # -------------------------------------------------------------
        # 1. CNN-Block (Feature Extraction)
        # -------------------------------------------------------------
        # Der Filter scannt immer 3 Kerzen (kernel_size=3) gleichzeitig
        # auf der Suche nach Mikro-Mustern (wie Ausbrüchen oder Dojis).
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(TIME_STEPS, FEATURES)),
        
        # Reduziert das Rauschen und komprimiert die Datenpunkte auf die stärksten Signale
        MaxPooling1D(pool_size=2),
        
        # -------------------------------------------------------------
        # 2. LSTM-Block (Temporal Sequence)
        # -------------------------------------------------------------
        # Verarbeitet die zeitliche Reihenfolge der vom CNN gefundenen Muster.
        LSTM(50, return_sequences=False),
        
        # Verhindert Overfitting (Auswendiglernen) durch Deaktivierung von 20% der Neuronen
        Dropout(0.2),
        
        # -------------------------------------------------------------
        # 3. Output-Block (Klassifizierung)
        # -------------------------------------------------------------
        # Sigmoid-Aktivierung gibt eine Wahrscheinlichkeit zwischen 0 und 1 zurück
        Dense(1, activation='sigmoid')
    ])
    
    # Kompilierung für Klassifikations-Aufgaben
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Modell initialisieren und Architektur ausgeben
trading_bot_model = create_cnnlstm_model()
trading_bot_model.summary()