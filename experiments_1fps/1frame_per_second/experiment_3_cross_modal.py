#!/usr/bin/env python3
"""
Experiment 3: Cross-Modal Prediction - STANDARDIZED VERSION
==========================================================
Building on Experiments 1&2: Video + Annotation A → Predict Annotation C
Architecture: SmallFastCNN(video) + MLP(annotation) → Concatenate → Predictor
Goal: Learn cross-modal relationships between different annotation types

Key Standardizations from Experiment 1:
- Every 5th frame sampling (0.2 FPS) for efficiency
- Same preprocessing pipeline and data loading
- Same SmallFastCNN base architecture
- Same training protocols and optimization
- Clean subfolder organization

Cross-Modal Learning Strategy:
- Physical→Behavioral: Use physical markers to predict head movements
- Behavioral→Physical: Use head movements to predict emotional states
- Test strategic combinations for proof of concept
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import json
import time
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# --- Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_optimal_device():
    """Checks for available hardware and returns the best torch device."""
    if torch.cuda.is_available():
        print("CUDA device found.")
        return torch.device("cuda:1")  # Use GPU 1 for 1fps experiments
    elif torch.backends.mps.is_available():
        print("MPS device found.")
        return torch.device("mps")
    else:
        print("No GPU found, using CPU.")
        return torch.device("cpu")

# --- Configuration (Standardized with Experiment 1) ---
BASE_DIR = "/home/tjeei/MultiModal Collab/1frame_per_second"
CONFIG = {
    "fps": 1.0,  # 1 frame per second for higher temporal resolution
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 15,
    "learning_rate": 0.001,
    "train_split": 0.8,
    "num_workers": os.cpu_count(),
    "device": get_optimal_device(),
    "use_amp": True,
    "data_dir": "/home/tjeei/MultiModal Project",  # Remote WSL data location
    "base_dir": BASE_DIR,
    "results_file": f"{BASE_DIR}/experiment_3_1fps/results/experiment_3_1fps_results.json",
    "padding_strategy": "repeat_last",
    "frame_cache_dir": f"{BASE_DIR}/experiment_3_1fps/frame_cache",
    "metadata_cache_file": f"{BASE_DIR}/experiment_3_1fps/processed_data_cache.pkl",
    "checkpoint_dir": f"{BASE_DIR}/experiment_3_1fps/checkpoints",
    "log_dir": f"{BASE_DIR}/experiment_3_1fps/logs",
    "model_dir": f"{BASE_DIR}/experiment_3_1fps/models",
    "force_recache": False,
    "early_stopping_patience": 5,
    # Optimized preprocessing options
    "use_optimized_preprocessing": True,
    "preprocessed_data_dir": f"{BASE_DIR}/preprocessed_data_1fps",
    "force_reprocess": False
}

# 14 Emotion markers
EMOTIONS = [
    "Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise",
    "Engagement", "Positive Valence", "Negative Valence", "Neutral Valence",
    "Sentimentality", "Confusion", "Neutral"
]

# 33 Behavior markers (21 Facial + 12 Behavioral)
BEHAVIORS = [
    # Facial Expression (21)
    "Attention", "Brow Furrow", "Brow Raise", "Cheek Raise", "Chin Raise",
    "Dimpler", "Eye Closure", "Eye Widen", "Inner Brow Raise", "Jaw Drop",
    "Lip Corner Depressor", "Lip Press", "Lip Pucker", "Lip Stretch", "Lip Suck",
    "Lip Tighten", "Mouth Open", "Nose Wrinkle", "Smile", "Smirk", "Upper Lip Raise",
    # Behavioral (12)
    "Head Pointing Up", "Head Pointing Down", "Head Pointing Forward", 
    "Head Turned Right", "Head Turned Left", "Head Turned Forward",
    "Head Tilted Left", "Head Tilted Right", "Head Not Tilted",
    "Head Leaning Forward", "Head Leaning Backward", "Lip Tuck"
]

# Cross-modal emotion↔behavior pairs for psychological research
EMOTION_BEHAVIOR_PAIRS = [
    # Emotion → Behavior (test if emotions predict behaviors)
    {"input_marker": "Joy", "output_marker": "Smile", "direction": "emotion_to_behavior"},
    {"input_marker": "Anger", "output_marker": "Brow Furrow", "direction": "emotion_to_behavior"},
    {"input_marker": "Sadness", "output_marker": "Lip Corner Depressor", "direction": "emotion_to_behavior"},
    {"input_marker": "Surprise", "output_marker": "Brow Raise", "direction": "emotion_to_behavior"},
    {"input_marker": "Disgust", "output_marker": "Nose Wrinkle", "direction": "emotion_to_behavior"},
    {"input_marker": "Fear", "output_marker": "Eye Widen", "direction": "emotion_to_behavior"},
    
    # Behavior → Emotion (test if behaviors predict emotions)
    {"input_marker": "Smile", "output_marker": "Joy", "direction": "behavior_to_emotion"},
    {"input_marker": "Brow Furrow", "output_marker": "Anger", "direction": "behavior_to_emotion"},
    {"input_marker": "Lip Corner Depressor", "output_marker": "Sadness", "direction": "behavior_to_emotion"},
    {"input_marker": "Brow Raise", "output_marker": "Surprise", "direction": "behavior_to_emotion"},
    {"input_marker": "Nose Wrinkle", "output_marker": "Disgust", "direction": "behavior_to_emotion"},
    {"input_marker": "Eye Widen", "output_marker": "Fear", "direction": "behavior_to_emotion"},
]

print("🚀 EXPERIMENT 3: CROSS-MODAL PREDICTION (1 FPS)")
print("="*70)
print(f"💻 Device: {CONFIG['device']}")
print(f"📊 Batch Size: {CONFIG['batch_size']}")
print(f"⚡ Automatic Mixed Precision: {CONFIG['use_amp']}")
print(f"🎯 FPS: {CONFIG['fps']} (1 frame per second)")
print(f"📈 Epochs: {CONFIG['num_epochs']}")
print(f"🧠 Model: SmallFastCNN + Cross-Modal Architecture")
print(f"⚙️ Emotion↔Behavior Pairs: {len(EMOTION_BEHAVIOR_PAIRS)}")
print(f"📁 Output Directory: {CONFIG['base_dir']}/experiment_3_1fps/")
print("="*70)

# --- DATASET ---

class CrossModalDataset(Dataset):
    """Dataset for cross-modal prediction: video + input_annotation → output_annotation"""
    def __init__(self, frame_paths, input_annotations, output_annotations):
        self.frame_paths = frame_paths
        self.input_annotations = input_annotations  # Input marker values
        self.output_annotations = output_annotations  # Target marker values
        
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        input_annotation = self.input_annotations[idx]
        output_annotation = self.output_annotations[idx]
        
        if frame_path.endswith('.npz'):
            frame = np.load(frame_path)['frame']
        elif frame_path.endswith('.npy'):
            frame = np.load(frame_path)
        else:
            frame = cv2.imread(frame_path)
            if frame is None:
                frame = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)
            frame = cv2.resize(frame, (CONFIG["img_size"], CONFIG["img_size"]))
        
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        
        return (torch.tensor(frame, dtype=torch.float32), 
                torch.tensor(input_annotation, dtype=torch.float32),
                torch.tensor(output_annotation, dtype=torch.float32))

class OptimizedCrossModalDataset(Dataset):
    """Optimized cross-modal dataset using preprocessed .npz files."""
    def __init__(self, preprocessed_files, input_marker, output_marker, all_markers):
        self.data = []
        self.input_marker_idx = all_markers.index(input_marker)
        self.output_marker_idx = all_markers.index(output_marker)
        
        # Load all data and create frame/annotation triplets
        for participant_file in preprocessed_files:
            data = np.load(participant_file)
            frames = data['frames']
            annotations = data['annotations']  # Shape: [num_frames, num_markers]
            
            # Extract input and output marker annotations
            input_annotations = annotations[:, self.input_marker_idx]
            output_annotations = annotations[:, self.output_marker_idx]
            
            # Add each frame with its input/output annotation pair
            for frame_idx in range(len(frames)):
                self.data.append({
                    'frame': frames[frame_idx],
                    'input_annotation': input_annotations[frame_idx],
                    'output_annotation': output_annotations[frame_idx],
                    'participant_file': participant_file,
                    'frame_idx': frame_idx
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frame = item['frame'].astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        input_annotation = item['input_annotation']
        output_annotation = item['output_annotation']
        
        return (torch.tensor(frame, dtype=torch.float32),
                torch.tensor(input_annotation, dtype=torch.float32),
                torch.tensor(output_annotation, dtype=torch.float32))

# Copy optimized preprocessing functions from experiment_1
def preprocess_all_data_once(config, video_locations, all_markers):
    """One-time preprocessing that saves all frames + annotations for all experiments."""
    preprocessed_data_dir = config["preprocessed_data_dir"]
    
    # Check if preprocessed data already exists
    if os.path.exists(preprocessed_data_dir) and len(os.listdir(preprocessed_data_dir)) > 0 and not config.get("force_reprocess", False):
        print(f"⚡️ Loading pre-processed data from {preprocessed_data_dir}")
        return load_preprocessed_data(preprocessed_data_dir, all_markers)
    
    print(f"🔄 One-time preprocessing: videos + annotations (FPS: {config['fps']})...")
    print("📝 This will save processed data for ALL experiments to avoid redundant processing")
    
    # Load annotation data
    big_interview_annotations = pd.read_csv(os.path.join(config["data_dir"], "Big Interview/big_interview_annotations.csv"))
    genex_annotations = pd.read_csv(os.path.join(config["data_dir"], "Genex/genex_annotations.csv"))
    
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    processed_data = []
    
    for video_dir in video_locations:
        for video_file in os.listdir(video_dir):
            if not video_file.lower().endswith(('.mp4', '.wmv', '.avi')):
                continue
            
            video_path = os.path.join(video_dir, video_file)
            participant_id = video_file.split('_')[1].split('.')[0]
            
            # Get appropriate annotations
            if "Big Interview" in video_dir:
                participant_annotations = big_interview_annotations[
                    big_interview_annotations['Respondent Name'] == participant_id
                ]
            else:
                participant_annotations = genex_annotations[
                    genex_annotations['Respondent Name'] == participant_id
                ]
            
            if participant_annotations.empty:
                continue
            
            print(f"  Processing: {participant_id} ({len(participant_annotations)} annotations)")
            
            # Process video and extract frames
            frames = extract_frames_from_video(video_path, config["fps"], config["img_size"])
            
            if len(frames) == 0:
                continue
            
            # Create frame-level labels
            labels = np.zeros((len(frames), len(all_markers)), dtype=np.float32)
            marker_to_idx = {marker: idx for idx, marker in enumerate(all_markers)}
            
            for _, annotation in participant_annotations.iterrows():
                start_frame = int(annotation['Start Time (ms)'] / 1000 * config["fps"])
                end_frame = int(annotation['End Time (ms)'] / 1000 * config["fps"])
                marker = annotation['Marker Name']
                
                if marker in marker_to_idx:
                    marker_idx = marker_to_idx[marker]
                    for frame_idx in range(max(0, start_frame), min(len(frames), end_frame + 1)):
                        labels[frame_idx, marker_idx] = 1.0
            
            # Save participant data
            participant_file = os.path.join(preprocessed_data_dir, f"{participant_id}.npz")
            np.savez_compressed(
                participant_file,
                frames=np.array(frames, dtype=np.uint8),
                annotations=labels,
                frame_ids=np.arange(len(frames)),
                participant_id=participant_id,
                video_path=video_path,
                markers=all_markers
            )
            
            processed_data.append({
                "participant_id": participant_id,
                "file_path": participant_file,
                "num_frames": len(frames),
                "num_annotations": len(participant_annotations)
            })
    
    print(f"✅ One-time preprocessing complete! Saved {len(processed_data)} participants to {preprocessed_data_dir}")
    return load_preprocessed_data(preprocessed_data_dir, all_markers)

def extract_frames_from_video(video_path, fps, img_size):
    """Extract frames from video at specified FPS."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return []
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames

def load_preprocessed_data(preprocessed_data_dir, all_markers):
    """Load preprocessed data from .npz files."""
    processed_data = []
    
    for filename in os.listdir(preprocessed_data_dir):
        if filename.endswith('.npz'):
            participant_id = filename.replace('.npz', '')
            file_path = os.path.join(preprocessed_data_dir, filename)
            
            # Load to get metadata
            data = np.load(file_path)
            processed_data.append({
                "participant_id": participant_id,
                "file_path": file_path,
                "num_frames": len(data['frames'])
            })
    
    return {"data": processed_data}

# --- MODEL ARCHITECTURE ---

class SmallFastCNN(nn.Module):
    """Standardized Small & Fast CNN backbone (same as Experiment 1)"""
    def __init__(self, img_size):
        super(SmallFastCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        conv_output_size = (img_size // 8) ** 2 * 48
        self.fc1 = nn.Linear(conv_output_size, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return x

class CrossModalNet(nn.Module):
    """Cross-modal network: Video + Input Annotation → Output Annotation"""
    def __init__(self, img_size):
        super(CrossModalNet, self).__init__()
        # Video processing branch
        self.video_backbone = SmallFastCNN(img_size)
        
        # Annotation processing branch
        self.annotation_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Fusion and prediction
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),  # video_features + annotation_features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, video, input_annotation):
        # Process video
        video_features = self.video_backbone(video)
        
        # Process input annotation
        annotation_features = self.annotation_mlp(input_annotation.unsqueeze(-1))
        
        # Fuse features
        fused_features = torch.cat([video_features, annotation_features], dim=1)
        
        # Predict output annotation
        output = self.fusion(fused_features)
        return output.squeeze()

# --- DATA PROCESSING (Reused from Experiment 1) ---

def process_video_with_caching(video_path, frame_cache_dir, fps, img_size):
    """Process video frames with caching (same as Experiment 1)"""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_cache_dir = os.path.join(frame_cache_dir, video_id)
    
    if os.path.exists(video_cache_dir) and len(os.listdir(video_cache_dir)) > 0:
        frame_files = sorted([f for f in os.listdir(video_cache_dir) if f.endswith('.npy')])
        return [os.path.join(video_cache_dir, f) for f in frame_files]
    
    os.makedirs(video_cache_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return []
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / fps))
    
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (img_size, img_size))
            frame_filename = os.path.join(video_cache_dir, f"frame_{len(frame_paths):04d}.npy")
            np.save(frame_filename, frame)
            frame_paths.append(frame_filename)
        
        frame_count += 1
    
    cap.release()
    return frame_paths

def save_processed_frames_and_annotations(config, video_locations, all_markers):
    """Save processed video frames and annotations to separate files for faster loading"""
    processed_frames_dir = os.path.join(config["base_dir"], "processed_frames")
    processed_annotations_file = os.path.join(config["base_dir"], "processed_annotations.pkl")
    
    # Check if already processed
    if os.path.exists(processed_frames_dir) and os.path.exists(processed_annotations_file) and not config["force_recache"]:
        print(f"⚡️ Loading pre-processed frames and annotations from cache")
        with open(processed_annotations_file, 'rb') as f:
            annotations_data = pickle.load(f)
        return processed_frames_dir, annotations_data
    
    print(f"🔄 Processing and saving videos + annotations (FPS: {config['fps']})...")
    
    # Load annotation data
    big_interview_annotations = pd.read_csv(os.path.join(config["data_dir"], "Big Interview/big_interview_annotations.csv"))
    genex_annotations = pd.read_csv(os.path.join(config["data_dir"], "Genex/genex_annotations.csv"))
    
    os.makedirs(processed_frames_dir, exist_ok=True)
    annotations_data = []
    
    for video_dir in video_locations:
        for video_file in os.listdir(video_dir):
            if not video_file.lower().endswith(('.mp4', '.wmv', '.avi')):
                continue
            
            video_path = os.path.join(video_dir, video_file)
            participant_id = video_file.split('_')[1].split('.')[0]
            
            # Get appropriate annotations
            if "Big Interview" in video_dir:
                participant_annotations = big_interview_annotations[
                    big_interview_annotations['Respondent Name'] == participant_id
                ]
            else:
                participant_annotations = genex_annotations[
                    genex_annotations['Respondent Name'] == participant_id
                ]
            
            if participant_annotations.empty:
                continue
            
            print(f"  Processing: {participant_id} ({len(participant_annotations)} annotations)")
            
            # Process video frames and save them
            participant_frames_dir = os.path.join(processed_frames_dir, participant_id)
            frame_files = process_and_save_video_frames(video_path, participant_frames_dir, config["fps"], config["img_size"])
            
            if not frame_files:
                continue
            
            # Create frame-level labels
            labels = {marker: [0.0] * len(frame_files) for marker in all_markers}
            
            for _, annotation in participant_annotations.iterrows():
                start_frame = int(annotation['Start Time (ms)'] / 1000 * config["fps"])
                end_frame = int(annotation['End Time (ms)'] / 1000 * config["fps"])
                marker = annotation['Marker Name']
                
                if marker in labels:
                    for frame_idx in range(max(0, start_frame), min(len(frame_files), end_frame + 1)):
                        labels[marker][frame_idx] = 1.0
            
            annotations_data.append({
                "participant_id": participant_id,
                "frame_files": frame_files,
                "labels": labels
            })
    
    # Save annotations data
    with open(processed_annotations_file, 'wb') as f:
        pickle.dump({
            "annotations_data": annotations_data,
            "markers": all_markers
        }, f)
    
    print(f"✅ Processed data saved to {processed_frames_dir} and {processed_annotations_file}")
    return processed_frames_dir, {"annotations_data": annotations_data, "markers": all_markers}

def process_and_save_video_frames(video_path, output_dir, fps, img_size):
    """Process video and save frames as compressed .npz files"""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npz')])
        return [os.path.join(output_dir, f) for f in frame_files]
    
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return []
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / fps))
    
    frame_files = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (img_size, img_size))
            frame_filename = os.path.join(output_dir, f"frame_{len(frame_files):04d}.npz")
            np.savez_compressed(frame_filename, frame=frame)
            frame_files.append(frame_filename)
        
        frame_count += 1
    
    cap.release()
    return frame_files

def preprocess_and_cache_data(config, video_locations, all_markers, cross_modal_pair):
    """Preprocess and cache data for specific cross-modal pair"""
    metadata_cache_file = config["metadata_cache_file"].replace('.pkl', f'_{cross_modal_pair["input_marker"].replace(" ", "_")}_to_{cross_modal_pair["output_marker"].replace(" ", "_")}.pkl')
    
    if os.path.exists(metadata_cache_file) and not config["force_recache"]:
        print(f"⚡️ Loading pre-processed data from {metadata_cache_file}")
        with open(metadata_cache_file, 'rb') as f:
            return pickle.load(f)
    
    input_marker = cross_modal_pair["input_marker"]
    output_marker = cross_modal_pair["output_marker"]
    
    print(f"🔄 Pre-processing for {input_marker} → {output_marker} (FPS: {config['fps']})...")
    
    # Load annotation data
    big_interview_annotations = pd.read_csv(os.path.join(config["data_dir"], "Big Interview/big_interview_annotations.csv"))
    genex_annotations = pd.read_csv(os.path.join(config["data_dir"], "Genex/genex_annotations.csv"))
    
    all_frame_paths = []
    all_input_annotations = []
    all_output_annotations = []
    
    for video_dir in video_locations:
        for video_file in os.listdir(video_dir):
            if not video_file.lower().endswith(('.mp4', '.wmv', '.avi')):
                continue
            
            video_path = os.path.join(video_dir, video_file)
            participant_id = video_file.split('_')[1].split('.')[0]
            
            # Get appropriate annotations
            if "Big Interview" in video_dir:
                participant_annotations = big_interview_annotations[
                    big_interview_annotations['Respondent Name'] == participant_id
                ]
            else:
                participant_annotations = genex_annotations[
                    genex_annotations['Respondent Name'] == participant_id
                ]
            
            if participant_annotations.empty:
                continue
            
            # Include participant regardless of which markers they have
            # Missing markers will default to 0.0 (same logic as experiments 1 & 2)
            
            print(f"  Processing: {participant_id} ({len(participant_annotations)} annotations)")
            
            frame_paths = process_video_with_caching(
                video_path, config["frame_cache_dir"], config["fps"], config["img_size"]
            )
            
            if not frame_paths:
                continue
            
            # Create frame-level labels for both markers (default to 0.0)
            input_labels = [0.0] * len(frame_paths)
            output_labels = [0.0] * len(frame_paths)
            
            # Mark annotations for any marker that exists
            for _, annotation in participant_annotations.iterrows():
                start_frame = int(annotation['Start Time (ms)'] / 1000 * config["fps"])
                end_frame = int(annotation['End Time (ms)'] / 1000 * config["fps"])
                marker = annotation['Marker Name']
                
                if marker == input_marker:
                    for frame_idx in range(max(0, start_frame), min(len(frame_paths), end_frame + 1)):
                        input_labels[frame_idx] = 1.0
                elif marker == output_marker:
                    for frame_idx in range(max(0, start_frame), min(len(frame_paths), end_frame + 1)):
                        output_labels[frame_idx] = 1.0
            
            # Add to dataset
            all_frame_paths.extend(frame_paths)
            all_input_annotations.extend(input_labels)
            all_output_annotations.extend(output_labels)
    
    # Save cache
    os.makedirs(os.path.dirname(metadata_cache_file), exist_ok=True)
    cache_payload = {
        "frame_paths": all_frame_paths,
        "input_annotations": all_input_annotations,
        "output_annotations": all_output_annotations,
        "input_marker": input_marker,
        "output_marker": output_marker
    }
    with open(metadata_cache_file, 'wb') as f:
        pickle.dump(cache_payload, f)
    
    print(f"✅ Cross-modal pre-processing complete. Data cached to {metadata_cache_file}")
    print(f"📊 Total frames: {len(all_frame_paths)}")
    print(f"📊 Input positive: {sum(all_input_annotations)}, Output positive: {sum(all_output_annotations)}")
    return cache_payload

# --- TRAINING FUNCTIONS ---

def save_checkpoint(state, checkpoint_file):
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    torch.save(state, checkpoint_file)

def load_checkpoint(model, optimizer, scaler, checkpoint_file, config):
    if not os.path.exists(checkpoint_file):
        return 0, 0.0
    
    print(f" 📂 Resuming from checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=config["device"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if config["use_amp"] and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_accuracy']
    return start_epoch, best_val_acc

def log_epoch(log_dir, pair_name, epoch_data):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"cross_modal_{pair_name}.csv")
    log_df = pd.DataFrame([epoch_data])
    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)

def save_cross_modal_outputs(model, val_loader, config, cross_modal_pair):
    """Save model predictions, logits, and ground truth for cross-modal analysis"""
    model.eval()
    
    all_logits = []
    all_predictions = []
    all_targets = []
    all_input_annotations = []
    all_frame_ids = []
    
    input_marker = cross_modal_pair["input_marker"]
    output_marker = cross_modal_pair["output_marker"]
    
    print(f"💾 Saving outputs for {input_marker} → {output_marker}...")
    
    with torch.no_grad():
        for batch_idx, (video, input_annotation, output_annotation) in enumerate(val_loader):
            video = video.to(config["device"])
            input_annotation = input_annotation.to(config["device"])
            output_annotation = output_annotation.to(config["device"])
            
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                logits = model(video, input_annotation)
            
            predictions = (torch.sigmoid(logits.float()) > 0.5).float()
            
            # Store batch data
            all_logits.append(logits.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(output_annotation.cpu().numpy())
            all_input_annotations.append(input_annotation.cpu().numpy())
            
            # Create frame IDs for this batch
            batch_frame_ids = [f"batch_{batch_idx}_frame_{i}" for i in range(video.size(0))]
            all_frame_ids.extend(batch_frame_ids)
    
    # Concatenate all data
    all_logits = np.concatenate(all_logits)
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_input_annotations = np.concatenate(all_input_annotations)
    
    # Save outputs
    outputs_dir = os.path.join(config["base_dir"], "model_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    safe_input = input_marker.replace(' ', '_').replace('/', '_')
    safe_output = output_marker.replace(' ', '_').replace('/', '_')
    output_file = os.path.join(outputs_dir, f"experiment_3_outputs_{safe_input}_to_{safe_output}.npz")
    
    np.savez_compressed(
        output_file,
        logits=all_logits,
        predictions=all_predictions,
        targets=all_targets,
        input_annotations=all_input_annotations,
        frame_ids=all_frame_ids,
        input_marker=input_marker,
        output_marker=output_marker,
        direction=cross_modal_pair["direction"]
    )
    
    print(f"✅ Cross-modal outputs saved to {output_file}")
    print(f"   Shape: {all_logits.shape} (samples,)")
    
    return output_file

def train_cross_modal_model(model, train_loader, val_loader, config, cross_modal_pair):
    # Calculate class weights for positive class optimization
    print("📊 Calculating class weights for positive class optimization...")
    total_samples = 0
    total_positives = 0
    
    for video, input_annotation, output_annotation in train_loader:
        total_samples += output_annotation.shape[0]
        total_positives += output_annotation.sum().item()
    
    positive_rate = total_positives / total_samples
    # Calculate positive class weight (higher weight for rarer positive class)
    pos_weight = max(1.0, min(10.0, (total_samples - total_positives) / total_positives)) if total_positives > 0 else 1.0
    
    print(f"🎯 Positive class rate: {positive_rate:.3f}")
    print(f"🎯 Positive class weight: {pos_weight:.2f}")
    
    # Use class-weighted BCE loss for positive class optimization
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(config["device"]))
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    
    input_marker = cross_modal_pair["input_marker"]
    output_marker = cross_modal_pair["output_marker"]
    pair_name = f"{input_marker.replace(' ', '_')}_to_{output_marker.replace(' ', '_')}"
    
    checkpoint_file = os.path.join(config["checkpoint_dir"], f"{pair_name}_checkpoint.pth")
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, scaler, checkpoint_file, config)
    
    print(f"🏋️ Training {input_marker} → {output_marker} from epoch {start_epoch} on {config['device']}...")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config["num_epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for video, input_annotation, output_annotation in train_pbar:
            video = video.to(config["device"])
            input_annotation = input_annotation.to(config["device"])
            output_annotation = output_annotation.to(config["device"])
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                outputs = model(video, input_annotation)
                loss = criterion(outputs, output_annotation.to(outputs.dtype))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for video, input_annotation, output_annotation in val_loader:
                video = video.to(config["device"])
                input_annotation = input_annotation.to(config["device"])
                output_annotation = output_annotation.to(config["device"])
                
                with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                    outputs = model(video, input_annotation)
                    loss = criterion(outputs, output_annotation.to(outputs.dtype))
                
                val_loss += loss.item()
                # Use optimized threshold for positive class detection
                preds = (torch.sigmoid(outputs.float()) > 0.4).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(output_annotation.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate comprehensive metrics for cross-modal prediction (positive class focused)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)  # Default is positive class F1
        val_precision = precision_score(val_targets, val_preds, zero_division=0)  # Positive class precision
        val_recall = recall_score(val_targets, val_preds, zero_division=0)  # Positive class recall
        
        # Additional positive class metrics
        pos_rate_target = np.mean(val_targets)
        pos_rate_pred = np.mean(val_preds)
        
        print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        print(f"    🎯 POSITIVE CLASS: Acc={val_acc:.4f} | F1={val_f1:.4f} | P={val_precision:.4f} | R={val_recall:.4f}")
        print(f"    🎯 Positive Rate: Target={pos_rate_target:.3f} | Pred={pos_rate_pred:.3f}")
        
        # Logging
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc,
            'val_f1_score': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'input_marker': input_marker,
            'output_marker': output_marker
        }
        log_epoch(config["log_dir"], pair_name, epoch_data)
        
        # Save checkpoint and best model (optimize for F1 instead of accuracy)
        if val_f1 > best_val_acc:
            best_val_acc = val_f1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if config["use_amp"] else None,
                'best_val_accuracy': best_val_acc,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_f1_score': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall
            }
            save_checkpoint(checkpoint_state, checkpoint_file)
            
            # Save best model
            os.makedirs(config["model_dir"], exist_ok=True)
            model_path = os.path.join(config["model_dir"], f"best_{pair_name}_model.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"  🛑 Early stopping triggered")
                break
    
    # Save model outputs for further analysis
    print(f"\n💾 Saving best cross-modal outputs...")
    output_file = save_cross_modal_outputs(model, val_loader, config, cross_modal_pair)
    
    # Final metrics to return
    final_f1 = val_f1
    final_precision = val_precision
    final_recall = val_recall
    
    return {
        "status": "completed",
        "best_val_accuracy": best_val_acc,
        "final_val_f1": final_f1,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "epochs_trained": epoch + 1,
        "input_marker": input_marker,
        "output_marker": output_marker,
        "direction": cross_modal_pair["direction"],
        "output_file": output_file
    }

# --- MAIN FUNCTION ---

def main():
    """Main execution function for cross-modal training"""
    video_locations = [
        "/home/tjeei/MultiModal Project/Big Interview/Just Face",
        "/home/tjeei/MultiModal Project/Genex/Just Respondent"
    ]
    
    # Load markers
    markers_file = os.path.join(CONFIG["data_dir"], "markers_47.txt")
    with open(markers_file, "r") as f:
        all_markers = [line.strip() for line in f if line.strip()]
    
    print(f"✅ {len(all_markers)} markers loaded.")
    
    # Train each cross-modal pair
    all_results = {}
    total_start_time = time.time()
    
    for i, cross_modal_pair in enumerate(EMOTION_BEHAVIOR_PAIRS):
        input_marker = cross_modal_pair["input_marker"]
        output_marker = cross_modal_pair["output_marker"]
        direction = cross_modal_pair["direction"]
        
        print(f"\\n" + "="*70)
        print(f"🎯 EMOTION↔BEHAVIOR PAIR {i+1}/{len(EMOTION_BEHAVIOR_PAIRS)}: {input_marker} → {output_marker}")
        print(f"📋 Direction: {direction}")
        print("="*70)
        
        # Load or create preprocessed data (first time only)
        if CONFIG["use_optimized_preprocessing"]:
            print("🚀 Using optimized preprocessing approach...")
            if i == 0:  # Only run preprocessing once for all pairs
                cached_data = preprocess_all_data_once(CONFIG, video_locations, all_markers)
                
                # Get list of preprocessed files
                preprocessed_files = []
                preprocessed_data_dir = CONFIG["preprocessed_data_dir"]
                if os.path.exists(preprocessed_data_dir):
                    preprocessed_files = [
                        os.path.join(preprocessed_data_dir, f) 
                        for f in os.listdir(preprocessed_data_dir) 
                        if f.endswith('.npz')
                    ]
                print(f"📊 Found {len(preprocessed_files)} preprocessed participant files")
            
            # Create optimized dataset for this cross-modal pair
            if preprocessed_files:
                print(f"⚡ Using OptimizedCrossModalDataset for {input_marker} → {output_marker}")
                dataset = OptimizedCrossModalDataset(preprocessed_files, input_marker, output_marker, all_markers)
                
                # Check for sufficient data from the dataset
                input_positive = sum(1 for i in range(len(dataset)) if dataset.data[i]['input_annotation'] > 0.5)
                output_positive = sum(1 for i in range(len(dataset)) if dataset.data[i]['output_annotation'] > 0.5)
            else:
                print("📼 Fallback to legacy approach - no preprocessed files found")
                cached_data = preprocess_and_cache_data(CONFIG, video_locations, all_markers, cross_modal_pair)
                dataset = CrossModalDataset(cached_data["frame_paths"], cached_data["input_annotations"], cached_data["output_annotations"])
                input_positive = sum(cached_data["input_annotations"])
                output_positive = sum(cached_data["output_annotations"])
        else:
            print("📼 Using legacy preprocessing approach...")
            cached_data = preprocess_and_cache_data(CONFIG, video_locations, all_markers, cross_modal_pair)
            dataset = CrossModalDataset(cached_data["frame_paths"], cached_data["input_annotations"], cached_data["output_annotations"])
            input_positive = sum(cached_data["input_annotations"])
            output_positive = sum(cached_data["output_annotations"])
        
        # Check for sufficient data
        if input_positive < 10 or output_positive < 10:
            print(f"⚠️ Skipping pair - insufficient data (Input: {int(input_positive)}, Output: {int(output_positive)})")
            all_results[f"{input_marker}_to_{output_marker}"] = {
                "status": "skipped",
                "reason": "insufficient_data",
                "input_positive": int(input_positive),
                "output_positive": int(output_positive)
            }
            continue
        
        # Train/val split
        train_size = int(CONFIG["train_split"] * len(dataset))
        val_size = len(dataset) - train_size
        g = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                                shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], 
                              shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        
        print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        # Create model
        model = CrossModalNet(CONFIG["img_size"]).to(CONFIG["device"])
        
        # Train model
        pair_start_time = time.time()
        results = train_cross_modal_model(model, train_loader, val_loader, CONFIG, cross_modal_pair)
        pair_time = (time.time() - pair_start_time) / 3600
        
        results["training_time_hours"] = pair_time
        all_results[f"{input_marker}_to_{output_marker}"] = results
        
        print(f"✅ Pair completed in {pair_time:.2f} hours (Accuracy: {results['best_val_accuracy']:.4f})")
    
    # Save final results
    total_time = (time.time() - total_start_time) / 3600
    final_results = {
        "experiment": "cross_modal_prediction",
        "total_training_time_hours": total_time,
        "cross_modal_pairs": all_results,
        "config": CONFIG
    }
    
    os.makedirs(os.path.dirname(CONFIG["results_file"]), exist_ok=True)
    with open(CONFIG["results_file"], "w") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\\n🎉 EXPERIMENT 3 (1 FPS) COMPLETE!")
    print(f"⏱️ Total time: {total_time:.2f} hours")
    print(f"📊 Cross-modal pairs trained: {len([r for r in all_results.values() if r.get('status') == 'completed'])}")
    print(f"📁 Results saved to: {CONFIG['results_file']}")
    print(f"📁 Models saved to: {CONFIG['model_dir']}")

if __name__ == "__main__":
    main()