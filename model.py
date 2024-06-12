import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pickle
import librosa

# Function to extract metadata
def getmetadata(filename):
    y, sr = librosa.load(filename)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    metadata_dict = {'tempo': tempo, 'chroma_stft': np.mean(chroma_stft), 'rmse': np.mean(rmse),
                     'spectral_centroid': np.mean(spec_centroid), 'spectral_bandwidth': np.mean(spec_bw),
                     'rolloff': np.mean(spec_rolloff), 'zero_crossing_rates': np.mean(zero_crossing)}

    for i in range(1, 21):
        metadata_dict.update({'mfcc' + str(i): np.mean(mfcc[i - 1])})

    return list(metadata_dict.values())

# 2. Loading the dataset
df = pd.read_csv("music_classification.csv")
print(df.columns)  # Check the columns to find the correct label column name
X = df.iloc[:, 1:28]
y = df['class_name']  # Replace 'class_name' with the correct column name if needed

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 3. Preprocessing the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Training the models
# SVM
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Logistic Regression
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_scaled, y_train)

# XGBoost
xgbc = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgbc.fit(X_train_scaled, y_train)

# 5. Saving the trained models using pickle
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm, file)

with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

with open('logreg_model.pkl', 'wb') as file:
    pickle.dump(logreg, file)

with open('xgbc_model.pkl', 'wb') as file:
    pickle.dump(xgbc, file)

# Save the scaler and label encoder as well
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

print("Models, scaler, and label encoder have been saved successfully.")
