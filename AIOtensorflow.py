import os
import sys
import numpy as np
import pandas as pd
import random

# Feat. Extraction
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft

#For hyperparameter tuning and ML models
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Labeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# For DL models (other than GraphCNN)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input, Reshape, Add, Layer
from tensorflow.keras.callbacks import EarlyStopping

# For GraphCNN (Spektral-based)
import scipy.sparse as sps
import spektral
from spektral.layers import GCNConv
from spektral.utils import sp_matrix_to_sp_tensor

###############################################
# Tee class to output to both terminal and file
###############################################
class Tee:
    def __init__(self, *files):
        self.files = files
        # Use encoding of the first file if available.
        self.encoding = getattr(files[0], "encoding", "utf-8")
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()
    def __getattr__(self, name):
        # Delegate attribute access to the first file.
        return getattr(self.files[0], name)

#############################################
# Global Configuration
#############################################
BANDS = {
    'delta':   [0.5, 4],
    'theta':   [4, 8],
    'alpha':   [8, 13],
    'beta':    [13, 30],
    'gamma':   [30, 45],
    'overall': [0.5, 45]
}
SAMPLING_RATE_EEG = 128  # Hz
SAMPLING_RATE_ECG = 256  # Hz (for multimodal)
NUM_PARTICIPANTS = 23
NUM_VIDEOS = 18

# New variable for FFT window size
FFT_WINDOW_SIZE = 128  # Change this to adjust the number of samples used in FFT

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

#############################################
# 1. Fuzzy and Defuzzification Layers
#############################################
class FuzzyLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.centers = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='uniform',
            trainable=True,
            name='centers'
        )
        self.spreads = self.add_weight(
            shape=(self.units,),
            initializer='uniform',
            trainable=True,
            name='spreads'
        )
        super(FuzzyLayer, self).build(input_shape)
    def call(self, inputs):
        # inputs: (batch, features)
        x_expanded = tf.expand_dims(inputs, axis=-1)  # -> (batch, features, 1)
        centers_exp = tf.expand_dims(self.centers, axis=0)
        spreads_exp = tf.expand_dims(self.spreads, axis=0)
        diff = x_expanded - centers_exp
        exponent = -tf.square(diff) / (2 * tf.square(spreads_exp))
        membership = tf.exp(exponent)  # -> (batch, features, units)
        return membership

class DefuzzLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Average across feature dimension: (batch, units)
        return tf.reduce_mean(inputs, axis=1)

#############################################
# 2. Real Gradient Reversal
#############################################
@tf.custom_gradient
def reverse_gradient(x):
    def grad(dy):
        return -dy
    return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def call(self, x):
        return reverse_gradient(x)

#############################################
# 3. FFT/Welch Feature Extraction Helpers
#############################################
def extract_features_fft(signal_data, band, sampling_rate=SAMPLING_RATE_EEG):
    # Use only the first FFT_WINDOW_SIZE samples if available
    # Ensure the signal is at least 2D
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)
    if signal_data.shape[0] > FFT_WINDOW_SIZE:
        signal_data = signal_data[:FFT_WINDOW_SIZE, :]
    low, high = band
    b, a = butter(4, [low/(sampling_rate/2), high/(sampling_rate/2)], btype='band')
    filtered = filtfilt(b, a, signal_data, axis=0)
    fft_vals = np.fft.rfft(filtered, axis=0)
    power = np.abs(fft_vals)**2
    features = np.sum(power, axis=0)
    return features

def extract_features_welch(signal_data, band, sampling_rate=SAMPLING_RATE_EEG):
    # Ensure the signal is at least 2D
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)
    low, high = band
    b, a = butter(4, [low/(sampling_rate/2), high/(sampling_rate/2)], btype='band')
    filtered = filtfilt(b, a, signal_data, axis=0)
    feats = []
    for ch in range(filtered.shape[1]):
        f, psd = welch(filtered[:, ch], fs=sampling_rate)
        idx = np.where((f >= low) & (f <= high))[0]
        feats.append(np.sum(psd[idx]))
    return np.array(feats)


#############################################
# 4. Emotion Classifier for Targeted Emotions
#############################################
def emotion_classifier(valence, arousal, dominance):
    """
    Classifies emotion by computing Euclidean distance from prototype means.
    Returns the predicted emotion name.
    """
    input_scores = np.array([valence, arousal, dominance])
    emotions = {
        "calmness":   {"mean": np.array([3.17, 2.26, 2.09])},
        "surprise":   {"mean": np.array([3.04, 3.00, 2.70])},
        "amusement":  {"mean": np.array([4.57, 3.83, 3.83])},
        "fear":       {"mean": np.array([2.04, 4.26, 4.13])},
        "excitement": {"mean": np.array([3.22, 3.70, 3.52])},
        "disgust":    {"mean": np.array([2.70, 3.83, 4.04])},
        "happiness":  {"mean": np.array([4.52, 3.17, 3.57])},
        "anger":      {"mean": np.array([1.35, 3.96, 4.35])},
        "sadness":    {"mean": np.array([1.39, 3.00, 3.48])}
    }
    min_dist = float('inf')
    pred = None
    for emo, proto in emotions.items():
        dist = np.linalg.norm(input_scores - proto["mean"])
        if dist < min_dist:
            min_dist = dist
            pred = emo
    return pred

#############################################
# 5. Data Loading & Preprocessing
#############################################
def load_and_preprocess_data(modality, feat_method, band_name, emotion_type):
    """
    Loads the DREAMER dataset and extracts features and labels.
      - modality: "EEG only" or "Multimodal" (if multimodal, only 'overall' is used)
      - feat_method: "FFT" or "Welch"
      - band_name: key from BANDS (for multimodal, only 'overall' is used)
      - emotion_type: one of "Valence", "Arousal", "Dominance", or "Targeted"
      
    For "Valence", "Arousal", and "Dominance", a median split is used to create binary labels.
    For "Targeted", the emotion_classifier is applied to obtain multi-class labels.
    
    Iterates over all EEG channels (assumed 14) and, for ECG in multimodal mode,
    extracts features from both channels and averages them.
    """
    data = loadmat('DREAMER.mat')
    X_list, y_list, groups = [], [], []
    
    for p in range(NUM_PARTICIPANTS):
        for v in range(NUM_VIDEOS):
            if modality == "EEG only":
                # Process all 14 EEG channels individually
                num_channels = 14
                features_per_video = []
                for ch in range(num_channels):
                    stim_ch = data['DREAMER'][0,0]['Data'][0, p]['EEG'][0,0]['stimuli'][0,0][v,0][:, ch]
                    base_ch = data['DREAMER'][0,0]['Data'][0, p]['EEG'][0,0]['baseline'][0,0][v,0][:, ch]
                    if feat_method == "FFT":
                        feat_stim = extract_features_fft(stim_ch, BANDS[band_name], SAMPLING_RATE_EEG)
                        feat_base = extract_features_fft(base_ch, BANDS[band_name], SAMPLING_RATE_EEG)
                    else:
                        feat_stim = extract_features_welch(stim_ch, BANDS[band_name], SAMPLING_RATE_EEG)
                        feat_base = extract_features_welch(base_ch, BANDS[band_name], SAMPLING_RATE_EEG)
                    # Each extraction returns a 1-element array; subtract and store the scalar.
                    features_per_video.append(feat_stim[0] - feat_base[0])
                features = np.array(features_per_video)
            else:
                # Multimodal: Process EEG (assumed multi-channel) and ECG (both channels)
                # EEG processing (using overall band)
                num_channels = 14
                features_per_video = []
                for ch in range(num_channels):
                    stim_ch = data['DREAMER'][0,0]['Data'][0, p]['EEG'][0,0]['stimuli'][0,0][v,0][:, ch]
                    base_ch = data['DREAMER'][0,0]['Data'][0, p]['EEG'][0,0]['baseline'][0,0][v,0][:, ch]
                    if feat_method == "FFT":
                        feat_stim = extract_features_fft(stim_ch, BANDS['overall'], SAMPLING_RATE_EEG)
                        feat_base = extract_features_fft(base_ch, BANDS['overall'], SAMPLING_RATE_EEG)
                    else:
                        feat_stim = extract_features_welch(stim_ch, BANDS['overall'], SAMPLING_RATE_EEG)
                        feat_base = extract_features_welch(base_ch, BANDS['overall'], SAMPLING_RATE_EEG)
                    # Each extraction returns a 1-element array; subtract and store the scalar.
                    features_per_video.append(feat_stim[0] - feat_base[0])
                eeg_features = np.array(features_per_video)
                # ECG processing: extract features separately from left and right channels and average them.
                ecg_stim = data['DREAMER'][0,0]['Data'][0, p]['ECG'][0,0]['stimuli'][0,0][v,0]
                ecg_base = data['DREAMER'][0,0]['Data'][0, p]['ECG'][0,0]['baseline'][0,0][v,0]
                # Expect ecg_stim and ecg_base to be 2D with 2 columns
                if ecg_stim.ndim == 1 or ecg_stim.shape[1] == 1:
                    raise Exception("ECG data does not have two channels")
                if feat_method == "FFT":
                    ecg_feat_stim_l = extract_features_fft(ecg_stim[:, 0], BANDS['overall'], SAMPLING_RATE_ECG)
                    ecg_feat_stim_r = extract_features_fft(ecg_stim[:, 1], BANDS['overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_l = extract_features_fft(ecg_base[:, 0], BANDS['overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_r = extract_features_fft(ecg_base[:, 1], BANDS['overall'], SAMPLING_RATE_ECG)
                else:
                    ecg_feat_stim_l = extract_features_welch(ecg_stim[:, 0], BANDS['overall'], SAMPLING_RATE_ECG)
                    ecg_feat_stim_r = extract_features_welch(ecg_stim[:, 1], BANDS['overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_l = extract_features_welch(ecg_base[:, 0], BANDS['overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_r = extract_features_welch(ecg_base[:, 1], BANDS['overall'], SAMPLING_RATE_ECG)
                # Average the two channels (each feature extraction returns a 1-element array)
                ecg_feature = ((ecg_feat_stim_l - ecg_feat_base_l) + (ecg_feat_stim_r - ecg_feat_base_r)) / 2

                features = np.concatenate([eeg_features, ecg_feature])
            
            X_list.append(features)
            
            # Retrieve scores for labels (remains unchanged)
            val = data['DREAMER'][0,0]['Data'][0, p]['ScoreValence'][0,0][v,0].astype(float)
            aro = data['DREAMER'][0,0]['Data'][0, p]['ScoreArousal'][0,0][v,0].astype(float)
            dom = data['DREAMER'][0,0]['Data'][0, p]['ScoreDominance'][0,0][v,0].astype(float)
            
            if emotion_type in ["Valence", "Arousal", "Dominance"]:
                if emotion_type == "Valence":
                    score = val
                elif emotion_type == "Arousal":
                    score = aro
                else:
                    score = dom
            else:  # Targeted
                score = emotion_classifier(val, aro, dom)
            y_list.append(score)
            groups.append(p)
    
    X = np.vstack(X_list)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # For Valence, Arousal, and Dominance use median split (binary classification)
    if emotion_type in ["Valence", "Arousal", "Dominance"]:
        med = np.median(np.array(y_list))
        y = np.where(np.array(y_list) >= med, 1, 0)
        num_classes = 2
    else:  # Targeted: multi-class
        le = LabelEncoder()
        y = le.fit_transform(np.array(y_list))
        num_classes = len(le.classes_)
    return X, y, np.array(groups), num_classes

#############################################
# 6. Classical Model Builders (ML)
#############################################
def build_svm_model(C=1.0, gamma=0.01):
    return SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced', probability=True)

def build_rf_model(n_estimators=100, max_depth=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced')

def build_dt_model(max_depth=None):
    return DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced')

#############################################
# 7. DL Model Builders
#############################################
def build_cnn_model(input_dim, num_classes=1):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)
    x = Conv1D(64, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    if num_classes == 1 or num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def build_fcnn_model(input_dim, num_classes=1, fuzzy_units=32):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)
    x = Conv1D(64, kernel_size=5, activation='relu')(x)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = FuzzyLayer(units=fuzzy_units)(x)
    x = BatchNormalization()(x)
    x = DefuzzLayer()(x) 
    x = Dense(100, activation='relu')(x)   
    x = Dropout(0.5)(x)
    if num_classes == 1 or num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def build_fuzzy_mha(input_dim, num_classes=1, fuzzy_units=64, num_heads=4, key_dim=16, conv1_filters=64, conv2_filters=32):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim,1))(inputs)
    x = Conv1D(conv1_filters, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(conv2_filters, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    fz = FuzzyLayer(units=fuzzy_units)(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(fz, fz)
    attn = Add()([fz, attn])
    attn = tf.keras.layers.LayerNormalization()(attn)
    defz = DefuzzLayer()(attn)
    bn = BatchNormalization()(defz)
    c = Dense(64, activation='relu')(bn)
    c = Dropout(0.5)(c)
    if num_classes == 1 or num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(c)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax')(c)
        loss = 'categorical_crossentropy'
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def build_domain_adv_fuzzy(input_dim, num_classes=1, fuzzy_units=64, conv1_filters=64, conv2_filters=32):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim,1))(inputs)
    x = Conv1D(conv1_filters, 3, activation='relu')(x)
    x = Conv1D(conv2_filters, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    fz = FuzzyLayer(units=fuzzy_units)(x)
    bn = BatchNormalization()(fz)
    defz = DefuzzLayer()(bn)
    # Emotion branch
    e = Dense(64, activation='relu')(defz)
    e = Dropout(0.5)(e)
    if num_classes == 1 or num_classes == 2:
        emo_out = Dense(1, activation='sigmoid', name='emo_out')(e)
        loss_emo = 'binary_crossentropy'
    else:
        emo_out = Dense(num_classes, activation='softmax', name='emo_out')(e)
        loss_emo = 'categorical_crossentropy'
    # Domain branch
    rev = GradientReversalLayer()(defz)
    s = Dense(32, activation='relu')(rev)
    subj_out = Dense(NUM_PARTICIPANTS, activation='softmax', name='subj_out')(s)
    model = Model(inputs=inputs, outputs=[emo_out, subj_out])
    model.compile(optimizer='adam', loss={'emo_out': loss_emo, 'subj_out': 'categorical_crossentropy'},
                  metrics={'emo_out': 'accuracy', 'subj_out': 'accuracy'})
    return model

#############################################
# GraphCNN components (adapted)
#############################################
class MyGCNConv(GCNConv):
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, self.channels])
    def call(self, inputs, **kwargs):
        x, a = inputs
        if not isinstance(a, tf.SparseTensor):
            a = tf.sparse.from_dense(a)
        return super().call([x, a], **kwargs)

class SampleAggregator(Layer):
    def __init__(self, num_channels=14, fuzzy_units=32, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.fuzzy = FuzzyLayer(units=fuzzy_units)
        self.defuzz = DefuzzLayer()
        self.bn = BatchNormalization()
    def call(self, inputs):
        n_nodes = tf.shape(inputs)[0]
        embed_dim = tf.shape(inputs)[-1]
        n_samp = n_nodes // self.num_channels
        shape3 = tf.stack([n_samp, self.num_channels, embed_dim])
        x_3d = tf.reshape(inputs, shape3)
        x_agg = tf.reduce_mean(x_3d, axis=1)
        mem = self.fuzzy(x_agg)
        crisp = self.defuzz(mem)
        crisp = self.bn(crisp)
        return crisp

def build_disjoint_gcn_fuzzy(embed_dim=32, num_classes=1, fuzzy_units=32, num_channels=14):
    x_in = Input(shape=(1,), name='X_in')
    a_in = Input((None,), name='A_in', sparse=True)
    gcn = MyGCNConv(embed_dim, activation='relu')([x_in, a_in])
    gcn = MyGCNConv(embed_dim, activation='relu')([gcn, a_in])
    aggregator = SampleAggregator(num_channels=num_channels, fuzzy_units=fuzzy_units)
    crisp = aggregator(gcn)
    if num_classes == 1 or num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(crisp)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax')(crisp)
        loss = 'categorical_crossentropy'
    model = Model(inputs=[x_in, a_in], outputs=outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'], run_eagerly=True)
    return model

#############################################
# 8. Evaluation Functions
#############################################
def evaluate_classical_model(model_builder, param_grid, X, y, groups, num_classes):
    logo = LeaveOneGroupOut()
    accs = []
    f1s = []
    grid = GridSearchCV(model_builder(), param_grid, cv=logo.split(X, y, groups), scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average='weighted'))
    return np.mean(accs)*100, np.mean(f1s)

def evaluate_dl_model(build_model_fn, X, y, groups, num_classes, epochs=50, batch_size=16):
    logo = LeaveOneGroupOut()
    accs = []
    f1s = []
    from tensorflow.keras.utils import to_categorical
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = build_model_fn(X.shape[1], num_classes)
        if isinstance(model.output, list):
            if num_classes == 2:
                y_train_target = y_train
                y_test_target = y_test
            else:
                y_train_target = to_categorical(y_train, num_classes=num_classes)
                y_test_target = to_categorical(y_test, num_classes=num_classes)
            subj_train = to_categorical(groups[train_idx], num_classes=NUM_PARTICIPANTS)
            subj_test = to_categorical(groups[test_idx], num_classes=NUM_PARTICIPANTS)
            model.fit(X_train, {'emo_out': y_train_target, 'subj_out': subj_train},
                      validation_data=(X_test, {'emo_out': y_test_target, 'subj_out': subj_test}),
                      epochs=epochs, batch_size=batch_size, verbose=0,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
            preds = model.predict(X_test)
            emo_preds = preds[0]
            if num_classes == 2:
                emo_preds = (emo_preds.flatten() > 0.5).astype(int)
            else:
                emo_preds = np.argmax(emo_preds, axis=1)
        else:
            if num_classes == 2:
                model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
                preds = model.predict(X_test)
                emo_preds = (preds.flatten() > 0.5).astype(int)
            else:
                y_train_cat = to_categorical(y_train, num_classes=num_classes)
                y_test_cat = to_categorical(y_test, num_classes=num_classes)
                model.fit(X_train, y_train_cat, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
                preds = model.predict(X_test)
                emo_preds = np.argmax(preds, axis=1)
        accs.append(accuracy_score(y_test, emo_preds))
        f1s.append(f1_score(y_test, emo_preds, average='weighted'))
    return np.mean(accs)*100, np.mean(f1s)

def evaluate_graph_model(build_graph_fn, X, y, groups, adjacency, num_classes, epochs=50):
    logo = LeaveOneGroupOut()
    accs = []
    f1s = []
    # Determine number of channels from X's feature dimension
    channels = X.shape[1]
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_graph = X_train.reshape((-1, channels, 1))
        X_test_graph  = X_test.reshape((-1, channels, 1))
        Xd_train, A_train, y_tr = make_disjoint_data(X_train_graph, 
            adjacency if channels==14 else (np.ones((channels,channels))-np.eye(channels)), y_train)
        Xd_test,  A_test, y_te = make_disjoint_data(X_test_graph, 
            adjacency if channels==14 else (np.ones((channels,channels))-np.eye(channels)), y_test)
        # Pass both channels and num_classes to the builder lambda
        model_gcn = build_graph_fn(num_channels=channels, num_classes=num_classes)
        loss = 'binary_crossentropy' if num_classes <= 2 else 'sparse_categorical_crossentropy'
        model_gcn.compile(optimizer='adam', loss=loss, metrics=['accuracy'], run_eagerly=True)
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        train_ds = single_batch_dataset(Xd_train, A_train, y_tr)
        test_ds  = single_batch_dataset(Xd_test, A_test, y_te)
        model_gcn.fit(train_ds, epochs=epochs, steps_per_epoch=1, verbose=0, callbacks=[es])
        preds = model_gcn.predict(test_ds, steps=1)
        
        # Handle predictions based on number of classes
        if num_classes <= 2:
            preds = (preds > 0.5).astype(int).flatten()
        else:
            preds = np.argmax(preds, axis=1)
            
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, average='weighted'))
    return np.mean(accs)*100, np.mean(f1s)

def my_block_diagonal(mats):
    sparse_mats = [sps.coo_matrix(m) for m in mats]
    return sps.block_diag(sparse_mats, format='coo')

def make_disjoint_data(X_eeg, adjacency, y):
    N = X_eeg.shape[0]
    A_list = [adjacency] * N
    A_block = my_block_diagonal(A_list)
    A_sp = sp_matrix_to_sp_tensor(A_block)
    X_list = [X_eeg[i] for i in range(N)]
    X_disjoint = np.vstack(X_list)
    return X_disjoint, A_sp, y

def single_batch_dataset(X_nodes, A_sp, y):
    ds = tf.data.Dataset.from_tensors(((X_nodes, A_sp), y))
    return ds

#############################################
# 9. Hyperparameter Tuning for DL Models
#############################################
def hyperparameter_tuning_dl(modality, feat_method, emotion_type):
    # Hyperparameter grids for DL models
    fcnn_params = [
        {'fuzzy_units': 32},
        {'fuzzy_units': 64}
    ]
    attention_params = [
        {'fuzzy_units': 32, 'num_heads': 2, 'key_dim': 16, 'conv1_filters': 64, 'conv2_filters': 32},
        {'fuzzy_units': 64, 'num_heads': 4, 'key_dim': 16, 'conv1_filters': 64, 'conv2_filters': 32},
        {'fuzzy_units': 32, 'num_heads': 2, 'key_dim': 16, 'conv1_filters': 64, 'conv2_filters': 128},
        {'fuzzy_units': 64, 'num_heads': 4, 'key_dim': 16, 'conv1_filters': 64, 'conv2_filters': 128},
    ]
    domain_adv_params = [
        {'fuzzy_units': 32, 'conv1_filters': 64, 'conv2_filters': 32},
        {'fuzzy_units': 64, 'conv1_filters': 64, 'conv2_filters': 32},
        {'fuzzy_units': 32, 'conv1_filters': 64, 'conv2_filters': 128},
        {'fuzzy_units': 64, 'conv1_filters': 64, 'conv2_filters': 128},
    ]
    graph_params = [
        {'embed_dim': 32, 'fuzzy_units': 32},
        {'embed_dim': 64, 'fuzzy_units': 32},
        {'embed_dim': 32, 'fuzzy_units': 64},
        {'embed_dim': 64, 'fuzzy_units': 64}
    ]

    
    # Loop over each frequency band
    for band in BANDS.keys():
        print(f"\n=== BAND: {band} ===")
        X, y, groups, num_classes = load_and_preprocess_data(modality, feat_method, band, emotion_type)
        print(f"Feature dimension: {X.shape[1]}, Number of classes: {num_classes}")

        
        # Hyperparameter tuning for FCNN (Fuzzy CNN)
        best_fcnn_acc = 0.0
        best_fcnn_conf = None
        for conf in fcnn_params:
            model_fn = lambda input_dim, num_classes: build_fcnn_model(input_dim, num_classes, fuzzy_units=conf['fuzzy_units'])
            acc_fcnn, _ = evaluate_dl_model(model_fn, X, y, groups, num_classes)
            print(f"FCNN config {conf} => Accuracy: {acc_fcnn:.2f}%")
            if acc_fcnn > best_fcnn_acc:
                best_fcnn_acc = acc_fcnn
                best_fcnn_conf = conf
        print(f"Best FCNN: Accuracy = {best_fcnn_acc:.2f}% with config {best_fcnn_conf}")
        
        # Hyperparameter tuning for FCNN+Attention
        best_att_acc = 0.0
        best_att_conf = None
        for conf in attention_params:
            model_fn = lambda input_dim, num_classes: build_fuzzy_mha(
                input_dim, num_classes,
                fuzzy_units=conf['fuzzy_units'],
                num_heads=conf['num_heads'],
                key_dim=conf['key_dim'],
                conv1_filters=conf['conv1_filters'],
                conv2_filters=conf['conv2_filters']
            )
            acc_att, _ = evaluate_dl_model(model_fn, X, y, groups, num_classes)
            print(f"FCNN+Attention config {conf} => Accuracy: {acc_att:.2f}%")
            if acc_att > best_att_acc:
                best_att_acc = acc_att
                best_att_conf = conf
        print(f"Best FCNN+Attention: Accuracy = {best_att_acc:.2f}% with config {best_att_conf}")
        
        # Hyperparameter tuning for Domain-Adversarial Fuzzy CNN
        best_da_acc = 0.0
        best_da_conf = None
        for conf in domain_adv_params:
            model_fn = lambda input_dim, num_classes: build_domain_adv_fuzzy(
                input_dim, num_classes,
                fuzzy_units=conf['fuzzy_units'],
                conv1_filters=conf['conv1_filters'],
                conv2_filters=conf['conv2_filters']
            )
            acc_da, _ = evaluate_dl_model(model_fn, X, y, groups, num_classes)
            print(f"Domain-Adversarial config {conf} => Accuracy: {acc_da:.2f}%")
            if acc_da > best_da_acc:
                best_da_acc = acc_da
                best_da_conf = conf
        print(f"Best Domain-Adversarial: Accuracy = {best_da_acc:.2f}% with config {best_da_conf}")
        
        # Hyperparameter tuning for GraphCNN
        best_graph_acc = 0.0
        best_graph_conf = None
        adjacency = np.ones((14, 14)) - np.eye(14)
        for conf in graph_params:
            model_fn = lambda: build_disjoint_gcn_fuzzy(
                embed_dim=conf['embed_dim'],
                num_classes=num_classes,
                fuzzy_units=conf['fuzzy_units']
            )
            acc_graph, _ = evaluate_graph_model(model_fn, X, y, groups, adjacency, num_classes)
            print(f"GraphCNN config {conf} => Accuracy: {acc_graph:.2f}%")
            if acc_graph > best_graph_acc:
                best_graph_acc = acc_graph
                best_graph_conf = conf
        print(f"Best GraphCNN: Accuracy = {best_graph_acc:.2f}% with config {best_graph_conf}")

###############################################
# 10. Main Loop with Full Grid Search & Logging
###############################################
def main():
    results = []
    modalities = ["EEG only", "Multimodal"] 
    feat_methods = ["Welch", "FFT"] 
    bands_dict = {
        "EEG only": ['delta','theta','alpha', 'beta','gamma','overall'],
        "Multimodal": ['overall']
    }
    emotion_types = ["Valence", "Arousal", "Dominance", "Targeted"] 
    model_names = ["SVM", "RF", "Decision Tree", "CNN", "FCNN", "FCNN+Attention", "Domain-Adversarial Fuzzy", "GraphCNN"]

    # Hyperparameter grids for classical models
    svm_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01]}
    rf_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
    dt_grid = {'max_depth': [None, 5, 10]}

    # Example adjacency for GraphCNN (assumed 14 nodes)
    adjacency = np.ones((14,14)) - np.eye(14)

    # Loop through all combinations in the given order
    for modality in modalities:
        for feat_method in feat_methods:
            for band in bands_dict[modality]:
                for emotion_type in emotion_types:
                    X, y, groups, num_classes = load_and_preprocess_data(modality, feat_method, band, emotion_type)
                    for model_name in model_names:
                        config_str = f"==={modality}, {feat_method}, {band}, {emotion_type}, {model_name}==="
                        print(config_str)
                        if model_name in ["SVM", "RF", "Decision Tree"]:
                            if model_name == "SVM":
                                mean_acc, mean_f1 = evaluate_classical_model(build_svm_model, svm_grid, X, y, groups, num_classes)
                            elif model_name == "RF":
                                mean_acc, mean_f1 = evaluate_classical_model(build_rf_model, rf_grid, X, y, groups, num_classes)
                            else:
                                mean_acc, mean_f1 = evaluate_classical_model(build_dt_model, dt_grid, X, y, groups, num_classes)
                        elif model_name in ["CNN", "FCNN", "FCNN+Attention", "Domain-Adversarial Fuzzy"]:
                            if model_name == "CNN":
                                mean_acc, mean_f1 = evaluate_dl_model(build_cnn_model, X, y, groups, num_classes)
                            elif model_name == "FCNN":
                                mean_acc, mean_f1 = evaluate_dl_model(build_fcnn_model, X, y, groups, num_classes)
                            elif model_name == "FCNN+Attention":
                                mean_acc, mean_f1 = evaluate_dl_model(build_fuzzy_mha, X, y, groups, num_classes)
                            elif model_name == "Domain-Adversarial Fuzzy":
                                mean_acc, mean_f1 = evaluate_dl_model(build_domain_adv_fuzzy, X, y, groups, num_classes)
                        elif model_name == "GraphCNN":
                            mean_acc, mean_f1 = evaluate_graph_model(
                                lambda num_channels, num_classes: build_disjoint_gcn_fuzzy(
                                    embed_dim=32, num_classes=num_classes, fuzzy_units=32, num_channels=num_channels),
                                X, y, groups, adjacency, num_classes)
                        else:
                            mean_acc, mean_f1 = 0, 0
                        result_str = f"Accuracy(percentage): {mean_acc:.2f}\nF1-Score: {mean_f1:.2f}\n"
                        print(result_str)
                        results.append((config_str, mean_acc, mean_f1))

    print("All results have been printed above.")

#################################################
# Redirect output to both terminal and result.txt
#################################################
if __name__ == '__main__':
    f = open("restex.txt", "w")
    sys.stdout = Tee(sys.stdout, f)
    # Run the main loop (if not already executed)
    main()   # (The above loop is executed on script run)
    # For hyperparameter tuning (on dl models, parameters can be changed)
    # The tuning function goes through all the bands, and it does not include normal CNN
    #hyperparameter_tuning_dl(modality="EEG only", feat_method="FFT", emotion_type="Targeted")
    sys.stdout = sys.__stdout__
    f.close()
