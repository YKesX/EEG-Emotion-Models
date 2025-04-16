import os
import sys
import numpy as np
import pandas as pd
import random
import argparse

# Feat. Extraction
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft

# For hyperparameter tuning and ML models
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Labeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# PyTorch imports (replacing TensorFlow)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

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
    'Delta':   [0.5, 4],
    'Theta':   [4, 8],
    'Alpha':   [8, 13],
    'Beta':    [13, 30],
    'Gamma':   [30, 45],
    'Overall': [0.5, 45]
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
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#############################################
# 1. Fuzzy and Defuzzification Layers
#############################################
class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, units=32):
        super(FuzzyLayer, self).__init__()
        self.units = units
        self.centers = nn.Parameter(torch.Tensor(input_dim, units))
        self.spreads = nn.Parameter(torch.Tensor(units))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.centers, -0.1, 0.1)
        nn.init.uniform_(self.spreads, 0.1, 1.0)
        
    def forward(self, x):
        # x: (batch, features)
        x_expanded = x.unsqueeze(-1)  # -> (batch, features, 1)
        centers_exp = self.centers.unsqueeze(0)  # -> (1, features, units)
        spreads_exp = self.spreads.unsqueeze(0).unsqueeze(1)  # -> (1, 1, units)
        
        diff = x_expanded - centers_exp
        exponent = -torch.pow(diff, 2) / (2 * torch.pow(spreads_exp, 2))
        membership = torch.exp(exponent)  # -> (batch, features, units)
        return membership

class DefuzzLayer(nn.Module):
    def __init__(self):
        super(DefuzzLayer, self).__init__()
        
    def forward(self, x):
        # Average across feature dimension: (batch, units)
        return torch.mean(x, dim=1)

#############################################
# 2. Real Gradient Reversal
#############################################
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation
    """
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GradientReversalLayer(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x)

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
def load_and_preprocess_data(modality, feat_method, band_name, emotion_type, dreamer_path='DREAMER.mat'):
    """
    Loads the DREAMER dataset and extracts features and labels.
      - modality: "EEG only" or "Multimodal" (if multimodal, only 'overall' is used)
      - feat_method: "FFT" or "Welch"
      - band_name: key from BANDS (for multimodal, only 'overall' is used)
      - emotion_type: one of "Valence", "Arousal", "Dominance", or "Targeted"
      - dreamer_path: Path to the DREAMER.mat file
      
    For "Valence", "Arousal", and "Dominance", a median split is used to create binary labels.
    For "Targeted", the emotion_classifier is applied to obtain multi-class labels.
    
    Iterates over all EEG channels (assumed 14) and, for ECG in multimodal mode,
    extracts features from both channels and averages them.
    """
    data = loadmat(dreamer_path)
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
                        feat_stim = extract_features_fft(stim_ch, BANDS['Overall'], SAMPLING_RATE_EEG)
                        feat_base = extract_features_fft(base_ch, BANDS['Overall'], SAMPLING_RATE_EEG)
                    else:
                        feat_stim = extract_features_welch(stim_ch, BANDS['Overall'], SAMPLING_RATE_EEG)
                        feat_base = extract_features_welch(base_ch, BANDS['Overall'], SAMPLING_RATE_EEG)
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
                    ecg_feat_stim_l = extract_features_fft(ecg_stim[:, 0], BANDS['Overall'], SAMPLING_RATE_ECG)
                    ecg_feat_stim_r = extract_features_fft(ecg_stim[:, 1], BANDS['Overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_l = extract_features_fft(ecg_base[:, 0], BANDS['Overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_r = extract_features_fft(ecg_base[:, 1], BANDS['Overall'], SAMPLING_RATE_ECG)
                else:
                    ecg_feat_stim_l = extract_features_welch(ecg_stim[:, 0], BANDS['Overall'], SAMPLING_RATE_ECG)
                    ecg_feat_stim_r = extract_features_welch(ecg_stim[:, 1], BANDS['Overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_l = extract_features_welch(ecg_base[:, 0], BANDS['Overall'], SAMPLING_RATE_ECG)
                    ecg_feat_base_r = extract_features_welch(ecg_base[:, 1], BANDS['Overall'], SAMPLING_RATE_ECG)
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
# PyTorch version of CNN
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(CNNModel, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output size after conv and pooling layers
        self.feature_size = self._get_conv_output_size(input_dim)
        
        # FC layers
        self.fc = nn.Linear(self.feature_size, 64)
        self.dropout = nn.Dropout(0.5)
        
        # Output layer
        if num_classes == 1 or num_classes == 2:
            self.output = nn.Linear(64, 1)
        else:
            self.output = nn.Linear(64, num_classes)
            
        self.num_classes = num_classes
        
    def _get_conv_output_size(self, shape):
        # Helper function to calculate the output size after conv layers
        x = torch.zeros(1, 1, shape)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return int(torch.flatten(x, 1).shape[1])
        
    def forward(self, x):
        # Reshape input: (batch, features) -> (batch, 1, features)
        x = x.view(x.size(0), 1, -1)
        
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        # Output
        x = self.output(x)
        if self.num_classes == 1 or self.num_classes == 2:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
            
        return x

# PyTorch version of Fuzzy CNN
class FCNNModel(nn.Module):
    def __init__(self, input_dim, num_classes=1, fuzzy_units=32):
        super(FCNNModel, self).__init__()
        self.num_classes = num_classes
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output size after conv and pooling layers
        self.feature_size = self._get_conv_output_size(input_dim)
        
        # Fuzzy layers
        self.fuzzy = FuzzyLayer(self.feature_size, fuzzy_units)
        self.bn = nn.BatchNorm1d(fuzzy_units)
        self.defuzz = DefuzzLayer()
        
        # FC layers
        self.fc = nn.Linear(fuzzy_units, 100)
        self.dropout = nn.Dropout(0.5)
        
        # Output layer
        if num_classes == 1 or num_classes == 2:
            self.output = nn.Linear(100, 1)
        else:
            self.output = nn.Linear(100, num_classes)
    
    def _get_conv_output_size(self, shape):
        # Helper function to calculate the output size after conv layers
        x = torch.zeros(1, 1, shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return int(torch.flatten(x, 1).shape[1])
    
    def forward(self, x):
        # Reshape input: (batch, features) -> (batch, 1, features)
        x = x.view(x.size(0), 1, -1)
        
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fuzzy layers
        x = self.fuzzy(x)  # Output: (batch, features, fuzzy_units)
        x = self.defuzz(x)  # Output: (batch, fuzzy_units)
        
        # FC layers
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        # Output
        x = self.output(x)
        if self.num_classes == 1 or self.num_classes == 2:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
            
        return x

# PyTorch version of Fuzzy Multi-Head Attention
class FuzzyMHAModel(nn.Module):
    def __init__(self, input_dim, num_classes=1, fuzzy_units=64, num_heads=4, key_dim=16, 
                 conv1_filters=64, conv2_filters=32):
        super(FuzzyMHAModel, self).__init__()
        self.num_classes = num_classes
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, conv1_filters, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output size after conv and pooling layers
        self.feature_size = self._get_conv_output_size(input_dim)
        
        # Fuzzy layer
        self.fuzzy = FuzzyLayer(self.feature_size, fuzzy_units)
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim=fuzzy_units, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(fuzzy_units)
        
        # Defuzzification
        self.defuzz = DefuzzLayer()
        
        # FC layers
        self.fc = nn.Linear(fuzzy_units, 64)
        self.dropout = nn.Dropout(0.5)
        
        # Output layer
        if num_classes == 1 or num_classes == 2:
            self.output = nn.Linear(64, 1)
        else:
            self.output = nn.Linear(64, num_classes)
    
    def _get_conv_output_size(self, shape):
        # Helper function to calculate the output size after conv layers
        x = torch.zeros(1, 1, shape)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return int(torch.flatten(x, 1).shape[1])
    
    def forward(self, x):
        # Reshape input: (batch, features) -> (batch, 1, features)
        x = x.view(x.size(0), 1, -1)
        
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fuzzy layer
        fz = self.fuzzy(x)
        
        # Multi-head attention
        # For MHA, we need shape (batch, seq_len, embed_dim), but our seq_len is 1
        # We'll reshape to (batch, 1, fuzzy_units) for MHA
        fz_reshaped = fz.mean(dim=1).unsqueeze(1)
        attn_output, _ = self.mha(fz_reshaped, fz_reshaped, fz_reshaped)
        
        # Residual connection and layer norm
        attn_output = attn_output + fz_reshaped
        attn_output = self.layer_norm(attn_output)
        
        # Defuzzification (squeeze out the seq_len dimension)
        attn_output = attn_output.squeeze(1)
        x = self.defuzz(fz)
        
        # FC layers
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        # Output
        x = self.output(x)
        if self.num_classes == 1 or self.num_classes == 2:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
            
        return x

# PyTorch version of Domain Adversarial Fuzzy CNN
class DomainAdvFuzzyModel(nn.Module):
    def __init__(self, input_dim, num_classes=1, fuzzy_units=64, conv1_filters=64, conv2_filters=32):
        super(DomainAdvFuzzyModel, self).__init__()
        self.num_classes = num_classes
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, conv1_filters, kernel_size=3)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output size after conv and pooling layers
        self.feature_size = self._get_conv_output_size(input_dim)
        
        # Fuzzy layers
        self.fuzzy = FuzzyLayer(self.feature_size, fuzzy_units)
        self.defuzz = DefuzzLayer()
        
        # Emotion branch
        self.emo_fc = nn.Linear(fuzzy_units, 64)
        self.dropout = nn.Dropout(0.5)
        if num_classes == 1 or num_classes == 2:
            self.emo_out = nn.Linear(64, 1)
        else:
            self.emo_out = nn.Linear(64, num_classes)
        
        # Subject branch with gradient reversal
        self.grl = GradientReversalLayer()
        self.subj_fc = nn.Linear(fuzzy_units, 32)
        self.subj_out = nn.Linear(32, NUM_PARTICIPANTS)
    
    def _get_conv_output_size(self, shape):
        # Helper function to calculate the output size after conv layers
        x = torch.zeros(1, 1, shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return int(torch.flatten(x, 1).shape[1])
    
    def forward(self, x):
        # Reshape input: (batch, features) -> (batch, 1, features)
        x = x.view(x.size(0), 1, -1)
        
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fuzzy layers
        fz = self.fuzzy(x)
        defz = self.defuzz(fz)
        
        # Emotion branch
        e = F.relu(self.emo_fc(defz))
        e = self.dropout(e)
        emo_out = self.emo_out(e)
        if self.num_classes == 1 or self.num_classes == 2:
            emo_out = torch.sigmoid(emo_out)
        else:
            emo_out = F.softmax(emo_out, dim=1)
        
        # Domain branch
        rev = self.grl(defz)
        s = F.relu(self.subj_fc(rev))
        subj_out = F.softmax(self.subj_out(s), dim=1)
        
        return emo_out, subj_out

#############################################
# GraphCNN components (PyTorch version)
#############################################
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, adj):
        # x: Node features [batch_size * num_nodes, in_channels]
        # adj: Adjacency matrix [batch_size * num_nodes, batch_size * num_nodes]
        
        # Apply linear transformation
        x = self.linear(x)
        
        # If adj is sparse, we need to use a different approach
        if isinstance(adj, torch.sparse.FloatTensor):
            # Message passing with sparse matrix
            x = torch.sparse.mm(adj, x)
        else:
            # Message passing with dense matrix
            x = torch.matmul(adj, x)
            
        return x

class SampleAggregator(nn.Module):
    def __init__(self, embed_dim=32, num_channels=14, fuzzy_units=32):
        super(SampleAggregator, self).__init__()
        self.num_channels = num_channels
        self.fuzzy = FuzzyLayer(input_dim=embed_dim, units=fuzzy_units)
        self.defuzz = DefuzzLayer()
        
    def forward(self, x):
        # x: [batch_size * num_nodes, embed_dim]
        batch_size = x.shape[0] // self.num_channels
        
        # Reshape to [batch_size, num_channels, embed_dim]
        x_3d = x.view(batch_size, self.num_channels, -1)
        
        # Aggregate across channels
        x_agg = torch.mean(x_3d, dim=1)  # [batch_size, embed_dim]
        
        # Apply fuzzy membership
        mem = self.fuzzy(x_agg)
        
        # Defuzzification
        crisp = self.defuzz(mem)
        
        return crisp

class DisjointGCNFuzzy(nn.Module):
    def __init__(self, embed_dim=32, num_classes=1, fuzzy_units=32, num_channels=14):
        super(DisjointGCNFuzzy, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # GCN layers
        self.gcn1 = GCNConv(1, embed_dim)
        self.gcn2 = GCNConv(embed_dim, embed_dim)
        
        # Aggregator
        self.aggregator = SampleAggregator(embed_dim=embed_dim, num_channels=num_channels, fuzzy_units=fuzzy_units)
        
        # Output layer
        if num_classes == 1 or num_classes == 2:
            self.output = nn.Linear(fuzzy_units, 1)
        else:
            self.output = nn.Linear(fuzzy_units, num_classes)
    
    def forward(self, x, adj):
        # x: [batch_size * num_nodes, 1]
        # adj: Sparse adjacency matrix
        
        # Apply GCN layers
        x = F.relu(self.gcn1(x, adj))
        x = F.relu(self.gcn2(x, adj))
        
        # Aggregate and apply fuzzy operations
        x = self.aggregator(x)
        
        # Output
        x = self.output(x)
        if self.num_classes == 1 or self.num_classes == 2:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
            
        return x

#############################################
# Dataset Utilities
#############################################
class EEGDataset(Dataset):
    def __init__(self, X, y, groups=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if isinstance(y[0], (int, float)) else torch.LongTensor(y)
        self.groups = torch.LongTensor(groups) if groups is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.groups is not None:
            return self.X[idx], self.y[idx], self.groups[idx]
        return self.X[idx], self.y[idx]

#############################################
# Graph Data Utilities
#############################################
def create_sparse_adjacency(adj_matrix):
    # Convert numpy adjacency matrix to PyTorch sparse tensor
    adj_matrix = adj_matrix.astype(np.float32)
    indices = torch.from_numpy(np.array(adj_matrix.nonzero()))
    values = torch.from_numpy(adj_matrix[adj_matrix.nonzero()])
    shape = torch.Size(adj_matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def create_block_diagonal(matrices):
    """Create block diagonal sparse matrix from a list of matrices"""
    n = len(matrices)
    if n == 0:
        return None
    
    # Get shapes and calculate offsets
    shapes = [m.shape for m in matrices]
    row_offsets = [0]
    col_offsets = [0]
    for i in range(n-1):
        row_offsets.append(row_offsets[-1] + shapes[i][0])
        col_offsets.append(col_offsets[-1] + shapes[i][1])
    
    total_rows = row_offsets[-1] + shapes[-1][0]
    total_cols = col_offsets[-1] + shapes[-1][1]
    
    # Collect all indices and values
    all_indices = []
    all_values = []
    
    for i, matrix in enumerate(matrices):
        # Get row and column indices of non-zero elements
        row_indices, col_indices = np.nonzero(matrix)
        
        # Adjust indices based on offsets
        row_indices += row_offsets[i]
        col_indices += col_offsets[i]
        
        # Stack row and column indices
        indices = np.vstack((row_indices, col_indices))
        
        # Get values
        values = matrix[row_indices - row_offsets[i], col_indices - col_offsets[i]]
        
        all_indices.append(indices)
        all_values.append(values)
    
    # Combine all indices and values
    if all_indices:
        combined_indices = np.hstack(all_indices)
        combined_values = np.hstack(all_values)
        
        # Convert to PyTorch tensors
        indices_tensor = torch.LongTensor(combined_indices)
        values_tensor = torch.FloatTensor(combined_values)
        
        # Create sparse tensor
        sparse_tensor = torch.sparse.FloatTensor(
            indices_tensor, values_tensor, torch.Size([total_rows, total_cols])
        )
        
        return sparse_tensor
    
    return None

def make_disjoint_data(X_eeg, adjacency, y):
    """
    Create disjoint graph data for batched processing
    
    Args:
        X_eeg: EEG data with shape [N, channels, features]
        adjacency: Adjacency matrix for channel connectivity
        y: Labels
        
    Returns:
        X_disjoint: Flattened node features
        A_sparse: Block diagonal sparse adjacency matrix
        y: Labels (unchanged)
    """
    N = X_eeg.shape[0]
    A_list = [adjacency] * N
    A_block = create_block_diagonal(A_list)
    
    # Flatten X_eeg to get node features
    X_list = [X_eeg[i] for i in range(N)]
    X_disjoint = np.vstack(X_list)
    
    return X_disjoint, A_block, y

# Custom collate function for graph data with sparse adjacency matrices
def graph_collate_fn(batch):
    # Since we're using a batch size of 1 for graph data, just return the first item
    # This avoids the need to batch sparse tensors which PyTorch doesn't support by default
    return batch[0]

class GraphDataset(Dataset):
    def __init__(self, X, adj, y):
        self.X = torch.FloatTensor(X)
        self.adj = adj  # Sparse tensor
        self.y = torch.FloatTensor(y) if isinstance(y[0], (int, float)) else torch.LongTensor(y)
        
    def __len__(self):
        return 1  # Single batch
    
    def __getitem__(self, idx):
        return self.X, self.adj, self.y

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

def evaluate_dl_model(model_class, X, y, groups, num_classes, epochs=50, batch_size=16, **model_kwargs):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logo = LeaveOneGroupOut()
    accs = []
    f1s = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        # Create PyTorch datasets
        train_dataset = EEGDataset(X_train, y_train, groups_train)
        test_dataset = EEGDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model with kwargs
        model = model_class(X.shape[1], num_classes, **model_kwargs).to(device)
        
        # Determine if model has domain adversarial component
        is_domain_adv = isinstance(model, DomainAdvFuzzyModel)
        
        # Loss function and optimizer
        if num_classes == 2 or num_classes == 1:
            criterion_emo = nn.BCELoss()
        else:
            criterion_emo = nn.CrossEntropyLoss()
            
        if is_domain_adv:
            criterion_subj = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(model.parameters())
        
        # Training loop
        best_val_loss = float('inf')
        best_weights = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                if is_domain_adv:
                    # For domain adversarial models, we need all three items (data, target, group)
                    inputs, targets, subject_ids = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    subject_ids = subject_ids.to(device)
                    
                    # Convert to one-hot for subject ids
                    subjects_one_hot = F.one_hot(subject_ids, num_classes=NUM_PARTICIPANTS).float()
                    
                    # Forward pass
                    emo_out, subj_out = model(inputs)
                    
                    # Prepare targets based on number of classes
                    if num_classes == 2 or num_classes == 1:
                        targets = targets.view(-1, 1).float()
                        loss_emo = criterion_emo(emo_out, targets)
                    else:
                        loss_emo = criterion_emo(emo_out, targets)
                    
                    loss_subj = criterion_subj(subj_out, subjects_one_hot)
                    loss = loss_emo + loss_subj
                else:
                    # For regular models, we need to handle the case where batch might have 3 items
                    # but we only care about the first two (data, target)
                    if len(batch) == 3:  # If batch contains group information
                        inputs, targets, _ = batch  # Ignore the group information
                    else:
                        inputs, targets = batch
                        
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Prepare targets based on number of classes
                    if num_classes == 2 or num_classes == 1:
                        targets = targets.view(-1, 1).float()
                        loss = criterion_emo(outputs, targets)
                    else:
                        loss = criterion_emo(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation (using training loss as proxy)
            avg_train_loss = train_loss / len(train_loader)
            
            # Early stopping
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best weights
        if best_weights is not None:
            model.load_state_dict(best_weights)
        
        # Evaluation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Extract just the inputs and targets, not the groups
                if len(batch) == 3:  # If batch contains group information
                    inputs, targets, _ = batch  # Ignore the group information
                else:
                    inputs, targets = batch
                    
                inputs, targets = inputs.to(device), targets.to(device)
                
                if is_domain_adv:
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)
                
                # Get predictions
                if num_classes == 2 or num_classes == 1:
                    preds = (outputs > 0.5).int().cpu().numpy()
                    preds = preds.flatten()
                else:
                    preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
        
        # Calculate metrics
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        accs.append(acc)
        f1s.append(f1)
    
    return np.mean(accs)*100, np.mean(f1s)

def evaluate_graph_model(model_class, X, y, groups, adjacency, num_classes, epochs=50, **model_kwargs):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logo = LeaveOneGroupOut()
    accs = []
    f1s = []
    
    # Determine number of channels from X's feature dimension
    channels = X.shape[1]
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Reshape and prepare graph data
        X_train_graph = X_train.reshape((-1, channels, 1))
        X_test_graph = X_test.reshape((-1, channels, 1))
        
        # Use fully connected graph if not using EEG channels
        if channels != 14:
            adj = np.ones((channels, channels)) - np.eye(channels)
        else:
            adj = adjacency
            
        # Create disjoint data
        Xd_train, A_train, y_tr = make_disjoint_data(X_train_graph, adj, y_train)
        Xd_test, A_test, y_te = make_disjoint_data(X_test_graph, adj, y_test)
        
        # Create datasets
        train_dataset = GraphDataset(Xd_train, A_train, y_tr)
        test_dataset = GraphDataset(Xd_test, A_test, y_te)
        
        # Create dataloaders with custom collate function to handle sparse tensors
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=graph_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=graph_collate_fn)
        
        # Initialize model
        model = model_class(num_channels=channels, num_classes=num_classes, **model_kwargs).to(device)
        
        # Loss function and optimizer
        if num_classes == 2 or num_classes == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(model.parameters())
        
        # Training loop
        best_loss = float('inf')
        best_weights = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for X_nodes, A_sp, targets in train_loader:
                X_nodes = X_nodes.to(device)
                A_sp = A_sp.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(X_nodes, A_sp)
                
                # Prepare targets based on number of classes
                if num_classes == 2 or num_classes == 1:
                    targets = targets.view(-1, 1).float()
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best weights
        if best_weights is not None:
            model.load_state_dict(best_weights)
        
        # Evaluation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_nodes, A_sp, targets in test_loader:
                X_nodes = X_nodes.to(device)
                A_sp = A_sp.to(device)
                
                outputs = model(X_nodes, A_sp)
                
                # Get predictions
                if num_classes == 2 or num_classes == 1:
                    preds = (outputs > 0.5).int().cpu().numpy()
                    preds = preds.flatten()
                else:
                    preds = outputs.argmax(dim=1).cpu().numpy()
                
                # Convert to list to ensure consistency with targets
                all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else preds)
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        accs.append(acc)
        f1s.append(f1)
    
    return np.mean(accs)*100, np.mean(f1s)

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
            acc_fcnn, _ = evaluate_dl_model(FCNNModel, X, y, groups, num_classes, 
                                          fuzzy_units=conf['fuzzy_units'])
            print(f"FCNN config {conf} => Accuracy: {acc_fcnn:.2f}%")
            if acc_fcnn > best_fcnn_acc:
                best_fcnn_acc = acc_fcnn
                best_fcnn_conf = conf
        print(f"Best FCNN: Accuracy = {best_fcnn_acc:.2f}% with config {best_fcnn_conf}")
        
        # Hyperparameter tuning for FCNN+Attention
        best_att_acc = 0.0
        best_att_conf = None
        for conf in attention_params:
            acc_att, _ = evaluate_dl_model(FuzzyMHAModel, X, y, groups, num_classes,
                                         fuzzy_units=conf['fuzzy_units'],
                                         num_heads=conf['num_heads'],
                                         key_dim=conf['key_dim'],
                                         conv1_filters=conf['conv1_filters'],
                                         conv2_filters=conf['conv2_filters'])
            print(f"FCNN+Attention config {conf} => Accuracy: {acc_att:.2f}%")
            if acc_att > best_att_acc:
                best_att_acc = acc_att
                best_att_conf = conf
        print(f"Best FCNN+Attention: Accuracy = {best_att_acc:.2f}% with config {best_att_conf}")
        
        # Hyperparameter tuning for Domain-Adversarial Fuzzy CNN
        best_da_acc = 0.0
        best_da_conf = None
        for conf in domain_adv_params:
            acc_da, _ = evaluate_dl_model(DomainAdvFuzzyModel, X, y, groups, num_classes,
                                        fuzzy_units=conf['fuzzy_units'],
                                        conv1_filters=conf['conv1_filters'],
                                        conv2_filters=conf['conv2_filters'])
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
            acc_graph, _ = evaluate_graph_model(DisjointGCNFuzzy, X, y, groups, adjacency, num_classes,
                                             embed_dim=conf['embed_dim'],
                                             fuzzy_units=conf['fuzzy_units'])
            print(f"GraphCNN config {conf} => Accuracy: {acc_graph:.2f}%")
            if acc_graph > best_graph_acc:
                best_graph_acc = acc_graph
                best_graph_conf = conf
        print(f"Best GraphCNN: Accuracy = {best_graph_acc:.2f}% with config {best_graph_conf}")

###############################################
# 10. Main Loop with Full Grid Search & Logging
###############################################
def main(args=None):
    # Parse command line arguments if provided
    if args is not None:
        modality = "EEG only" if args.modality == "eeg_only" else "Multimodal"
        feat_method = args.feat_method.upper()
        bands = args.bands.split(',')
        output_file = args.output
        selected_models = args.models.split(',') if args.models else ["SVM", "RF", "Decision Tree", "CNN", "FCNN", 
                                                                     "FCNN+Attention", "Domain-Adversarial Fuzzy", "GraphCNN"]

        # Validate modality and band combinations
        if modality == "Multimodal" and any(band != "Overall" for band in bands):
            print("Warning: For Multimodal, only 'Overall' band is supported. Ignoring other bands.")
            bands = ["Overall"]

        print(f"Running with modality: {modality}, feature method: {feat_method}")
        print(f"Selected bands: {', '.join(bands)}")
        print(f"Selected models: {', '.join(selected_models)}")
        print(f"Results will be saved to: {output_file}")

        results = []
        modalities = [modality]
        feat_methods = [feat_method]
        bands_dict = {
            "EEG only": bands,
            "Multimodal": ["Overall"]
        }
        model_names = selected_models
    else:
        # Use default configuration for complete run
        results = []
        modalities = ["EEG only", "Multimodal"]  
        feat_methods = ["Welch", "FFT"]  
        bands_dict = {
            "EEG only": ['Delta','Theta','Alpha', 'Beta','Gamma','Overall'],
            "Multimodal": ['Overall']
        }
        output_file = "pytorch_results.txt"
        model_names = ["SVM", "RF", "Decision Tree", "CNN", "FCNN", "FCNN+Attention", "Domain-Adversarial Fuzzy", "GraphCNN"]

    emotion_types = ["Valence","Arousal", "Dominance", "Targeted"]  

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
                        elif model_name == "CNN":
                            mean_acc, mean_f1 = evaluate_dl_model(CNNModel, X, y, groups, num_classes)
                        elif model_name == "FCNN":
                            mean_acc, mean_f1 = evaluate_dl_model(FCNNModel, X, y, groups, num_classes, fuzzy_units=32)
                        elif model_name == "FCNN+Attention":
                            mean_acc, mean_f1 = evaluate_dl_model(FuzzyMHAModel, X, y, groups, num_classes,
                                                                fuzzy_units=64, num_heads=4, key_dim=16, 
                                                                conv1_filters=64, conv2_filters=32)
                        elif model_name == "Domain-Adversarial Fuzzy":
                            mean_acc, mean_f1 = evaluate_dl_model(DomainAdvFuzzyModel, X, y, groups, num_classes,
                                                                fuzzy_units=64, conv1_filters=64, conv2_filters=32)
                        elif model_name == "GraphCNN":
                            mean_acc, mean_f1 = evaluate_graph_model(DisjointGCNFuzzy, X, y, groups, adjacency, num_classes,
                                                                  embed_dim=32, fuzzy_units=32)
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='EEG Emotion Recognition with PyTorch')
    parser.add_argument('--modality', choices=['eeg_only', 'multimodal'], 
                        default='eeg_only', help='Modality to use (eeg_only or multimodal)')
    parser.add_argument('--feat_method', choices=['fft', 'welch'],
                        default='fft', help='Feature extraction method (fft or welch)')
    parser.add_argument('--bands', type=str, default='Overall', 
                        help='Comma-separated list of bands to use (Delta,Theta,Alpha,Beta,Gamma,Overall)')
    parser.add_argument('--output', type=str, default='pytorch_results.txt',
                        help='Output file path for results')
    parser.add_argument('--models', type=str, default='',
                        help='Comma-separated list of models to evaluate (SVM,RF,Decision Tree,CNN,FCNN,FCNN+Attention,Domain-Adversarial Fuzzy,GraphCNN)')
    
    args = parser.parse_args()
    
    # Open output file and redirect stdout
    f = open(args.output, "w")
    sys.stdout = Tee(sys.stdout, f)
    
    # Run the main loop with arguments
    main(args)
    
    # Restore stdout and close file
    sys.stdout = sys.__stdout__
    f.close()
