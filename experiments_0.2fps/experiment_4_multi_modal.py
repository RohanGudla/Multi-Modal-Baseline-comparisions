#!/usr/bin/env python3
"""
Experiment 4: Multi-Input Multi-Output Cross-Modal - STANDARDIZED VERSION
========================================================================
Building on Experiments 1-3: Video + Multiple Annotations A,B,C → Predict Multiple Annotations X,Y,Z
Architecture: SmallFastCNN + Multiple Input MLPs → Fusion → Multiple Output MLPs
Goal: Advanced multi-modal learning with multiple inputs and outputs

Key Standardizations from Experiment 1:
- Every 5th frame sampling (0.2 FPS) for efficiency
- Same preprocessing pipeline and data loading
- Same SmallFastCNN base architecture
- Same training protocols and optimization
- Clean subfolder organization

Multi-Modal Strategy:
- Input: Video + 3 Physical markers → Output: 2 Behavioral markers
- Test ability to learn complex relationships
- Advanced fusion architecture for multiple signals
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
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS device found.")
        return torch.device("mps")
    else:
        print("No GPU found, using CPU.")
        return torch.device("cpu")

# --- Configuration (Standardized with Experiment 1) ---
BASE_DIR = "/home/tjeei/Multi Modal Collab"
CONFIG = {
    "fps": 0.2,  # STANDARDIZED: Every 5th frame (same as Exp 1)
    "img_size": 224,
    "batch_size": 24,  # Smaller batch for complex multi-modal architecture
    "num_epochs": 20,
    "learning_rate": 0.0008,  # Lower LR for complex multi-modal training
    "train_split": 0.8,
    "num_workers": os.cpu_count(),
    "device": get_optimal_device(),
    "use_amp": True,
    "data_dir": "/home/tjeei/MultiModal Project",  # Remote WSL data location
    "base_dir": BASE_DIR,
    "results_file": f"{BASE_DIR}/experiment_4/results/experiment_4_results.json",
    "padding_strategy": "repeat_last",
    "frame_cache_dir": f"{BASE_DIR}/experiment_4/frame_cache",
    "metadata_cache_file": f"{BASE_DIR}/experiment_4/processed_data_cache.pkl",
    "checkpoint_file": f"{BASE_DIR}/experiment_4/checkpoints/multi_modal_checkpoint.pth",
    "log_dir": f"{BASE_DIR}/experiment_4/logs",
    "model_dir": f"{BASE_DIR}/experiment_4/models",
    "force_recache": False,
    "early_stopping_patience": 7,
    # Optimized preprocessing options
    "use_optimized_preprocessing": True,
    "preprocessed_data_dir": f"{BASE_DIR}/preprocessed_data",
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

# Complete domain mapping configuration for psychological research
DOMAIN_MAPPING_CONFIG = {
    "emotion_to_behavior": {
        "input_markers": EMOTIONS,      # 14 emotions → 33 behaviors
        "output_markers": BEHAVIORS,    
        "input_dim": 14,
        "output_dim": 33,
        "experiment_name": "emotions_to_behaviors"
    },
    "behavior_to_emotion": {
        "input_markers": BEHAVIORS,     # 33 behaviors → 14 emotions  
        "output_markers": EMOTIONS,
        "input_dim": 33,
        "output_dim": 14,
        "experiment_name": "behaviors_to_emotions"
    }
}

print("🚀 EXPERIMENT 4: COMPLETE DOMAIN MAPPING (EMOTIONS↔BEHAVIORS)")
print("="*70)
print(f"💻 Device: {CONFIG['device']}")
print(f"📊 Batch Size: {CONFIG['batch_size']} (multi-modal optimized)")
print(f"⚡ Automatic Mixed Precision: {CONFIG['use_amp']}")
print(f"🎯 FPS: {CONFIG['fps']} (standardized - every 5th frame)")
print(f"📈 Epochs: {CONFIG['num_epochs']}")
print(f"🧠 Model: SmallFastCNN + Multi-Modal Fusion Architecture")
print(f"🧠 Emotions → Behaviors: {len(EMOTIONS)} → {len(BEHAVIORS)}")
print(f"🎭 Behaviors → Emotions: {len(BEHAVIORS)} → {len(EMOTIONS)}")
print(f"📁 Output Directory: {CONFIG['base_dir']}/experiment_4/")
print("="*70)

# --- DATASET ---

class MultiModalDataset(Dataset):
    """Dataset for multi-input multi-output prediction"""
    def __init__(self, frame_paths, input_annotations, output_annotations):
        self.frame_paths = frame_paths
        self.input_annotations = input_annotations    # List of input annotation vectors
        self.output_annotations = output_annotations  # List of output annotation vectors
        
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        input_annotations = self.input_annotations[idx]  # Vector of input marker values
        output_annotations = self.output_annotations[idx]  # Vector of output marker values
        
        if frame_path.endswith('.npy'):
            frame = np.load(frame_path)
        else:
            frame = cv2.imread(frame_path)
            if frame is None:
                frame = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)
            frame = cv2.resize(frame, (CONFIG["img_size"], CONFIG["img_size"]))
        
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        
        return (torch.tensor(frame, dtype=torch.float32),
                torch.tensor(input_annotations, dtype=torch.float32),
                torch.tensor(output_annotations, dtype=torch.float32))

class OptimizedMultiModalDataset(Dataset):
    """Optimized multi-modal dataset using preprocessed .npz files."""
    def __init__(self, preprocessed_files, input_markers, output_markers, all_markers):
        self.data = []
        
        # Get indices for input and output markers
        self.input_marker_indices = [all_markers.index(marker) for marker in input_markers if marker in all_markers]
        self.output_marker_indices = [all_markers.index(marker) for marker in output_markers if marker in all_markers]
        
        print(f"📊 Input markers found: {len(self.input_marker_indices)}/{len(input_markers)}")
        print(f"📊 Output markers found: {len(self.output_marker_indices)}/{len(output_markers)}")
        
        # Load all data and create frame/annotation tuples
        for participant_file in preprocessed_files:
            data = np.load(participant_file)
            frames = data['frames']
            annotations = data['annotations']  # Shape: [num_frames, num_markers]
            
            # Extract input and output marker annotations
            if len(self.input_marker_indices) > 0 and len(self.output_marker_indices) > 0:
                input_annotations = annotations[:, self.input_marker_indices]
                output_annotations = annotations[:, self.output_marker_indices]
                
                # Add each frame with its input/output annotation vectors
                for frame_idx in range(len(frames)):
                    self.data.append({
                        'frame': frames[frame_idx],
                        'input_annotations': input_annotations[frame_idx],
                        'output_annotations': output_annotations[frame_idx],
                        'participant_file': participant_file,
                        'frame_idx': frame_idx
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frame = item['frame'].astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        input_annotations = item['input_annotations']
        output_annotations = item['output_annotations']
        
        return (torch.tensor(frame, dtype=torch.float32),
                torch.tensor(input_annotations, dtype=torch.float32),
                torch.tensor(output_annotations, dtype=torch.float32))

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

class MultiModalNet(nn.Module):
    """Advanced multi-modal network: Video + Multiple Inputs → Multiple Outputs"""
    def __init__(self, img_size, num_input_markers, num_output_markers):
        super(MultiModalNet, self).__init__()
        self.num_input_markers = num_input_markers
        self.num_output_markers = num_output_markers
        
        # Video processing branch
        self.video_backbone = SmallFastCNN(img_size)
        
        # Multiple input annotation processing branches
        self.input_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16, 24),
                nn.ReLU()
            ) for _ in range(num_input_markers)
        ])
        
        # Fusion layer for all inputs
        fusion_input_size = 128 + (24 * num_input_markers)  # video + all input branches
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Multiple output prediction heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1)
            ) for _ in range(num_output_markers)
        ])
    
    def forward(self, video, input_annotations):
        # Process video
        video_features = self.video_backbone(video)
        
        # Process each input annotation separately
        input_features = []
        for i in range(self.num_input_markers):
            annotation_input = input_annotations[:, i:i+1]  # Select i-th marker
            features = self.input_branches[i](annotation_input)
            input_features.append(features)
        
        # Concatenate all features
        all_features = torch.cat([video_features] + input_features, dim=1)
        
        # Fusion
        fused_features = self.fusion_layer(all_features)
        
        # Generate multiple outputs
        outputs = []
        for head in self.output_heads:
            output = head(fused_features)
            outputs.append(output.squeeze())
        
        return torch.stack(outputs, dim=1)  # [batch_size, num_output_markers]

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

def preprocess_and_cache_data(config, video_locations, multi_modal_config):
    """Preprocess and cache data for multi-modal training"""
    metadata_cache_file = config["metadata_cache_file"]
    
    if os.path.exists(metadata_cache_file) and not config["force_recache"]:
        print(f"⚡️ Loading pre-processed data from {metadata_cache_file}")
        with open(metadata_cache_file, 'rb') as f:
            return pickle.load(f)
    
    input_markers = multi_modal_config["input_markers"]
    output_markers = multi_modal_config["output_markers"]
    
    print(f"🔄 Pre-processing for multi-modal training (FPS: {config['fps']})...")
    print(f"📥 Input markers: {input_markers}")
    print(f"📤 Output markers: {output_markers}")
    
    # Load annotation data
    big_interview_annotations = pd.read_csv(os.path.join(config["data_dir"], "Big Interview/big_interview_annotations.csv"))
    genex_annotations = pd.read_csv(os.path.join(config["data_dir"], "Genex/genex_annotations.csv"))
    
    all_frame_paths = []
    all_input_annotations = []  # List of input vectors
    all_output_annotations = []  # List of output vectors
    
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
            
            # Create frame-level labels for all markers
            all_markers = input_markers + output_markers
            labels = {marker: [0.0] * len(frame_paths) for marker in all_markers}
            
            for _, annotation in participant_annotations.iterrows():
                start_frame = int(annotation['Start Time (ms)'] / 1000 * config["fps"])
                end_frame = int(annotation['End Time (ms)'] / 1000 * config["fps"])
                marker = annotation['Marker Name']
                
                if marker in labels:
                    for frame_idx in range(max(0, start_frame), min(len(frame_paths), end_frame + 1)):
                        labels[marker][frame_idx] = 1.0
            
            # Add to dataset
            for i in range(len(frame_paths)):
                all_frame_paths.append(frame_paths[i])
                
                # Create input vector for this frame
                input_vector = [labels[marker][i] for marker in input_markers]
                all_input_annotations.append(input_vector)
                
                # Create output vector for this frame
                output_vector = [labels[marker][i] for marker in output_markers]
                all_output_annotations.append(output_vector)
    
    # Save cache
    os.makedirs(os.path.dirname(metadata_cache_file), exist_ok=True)
    cache_payload = {
        "frame_paths": all_frame_paths,
        "input_annotations": all_input_annotations,
        "output_annotations": all_output_annotations,
        "input_markers": input_markers,
        "output_markers": output_markers
    }
    with open(metadata_cache_file, 'wb') as f:
        pickle.dump(cache_payload, f)
    
    print(f"✅ Multi-modal pre-processing complete. Data cached to {metadata_cache_file}")
    print(f"📊 Total frames: {len(all_frame_paths)}")
    
    # Print statistics
    input_stats = np.array(all_input_annotations)
    output_stats = np.array(all_output_annotations)
    for i, marker in enumerate(input_markers):
        print(f"📥 {marker}: {input_stats[:, i].sum():.0f} positive samples")
    for i, marker in enumerate(output_markers):
        print(f"📤 {marker}: {output_stats[:, i].sum():.0f} positive samples")
    
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

def log_epoch(log_dir, epoch_data):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "multi_modal_training.csv")
    log_df = pd.DataFrame([epoch_data])
    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)

def train_multi_modal_model(model, train_loader, val_loader, config, multi_modal_config):
    # Calculate class weights for positive class optimization
    print("📊 Calculating class weights for positive class optimization...")
    pos_weights = []
    total_samples = 0
    total_positives = 0
    
    for video, input_annotations, output_annotations in train_loader:
        batch_pos = output_annotations.sum(dim=0)  # Positive count per output marker
        batch_total = output_annotations.shape[0]  # Total samples in batch
        pos_weights.append(batch_pos)
        total_samples += batch_total
        total_positives += output_annotations.sum().item()
    
    # Combine across all batches
    total_pos_per_marker = torch.stack(pos_weights).sum(dim=0)
    total_neg_per_marker = total_samples - total_pos_per_marker
    
    # Calculate positive class weights (higher weight for rarer positive class)
    pos_weight_tensor = total_neg_per_marker.float() / (total_pos_per_marker.float() + 1e-6)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=10.0)  # Reasonable bounds
    
    output_markers = multi_modal_config["output_markers"]
    print(f"🎯 Positive class rate: {total_positives/(total_samples*len(output_markers)):.3f}")
    print(f"🎯 Average pos_weight: {pos_weight_tensor.mean():.2f} (range: {pos_weight_tensor.min():.2f}-{pos_weight_tensor.max():.2f})")
    
    # Use class-weighted BCE loss for positive class optimization
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(config["device"]))
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, scaler, config["checkpoint_file"], config)
    
    input_markers = multi_modal_config["input_markers"]
    output_markers = multi_modal_config["output_markers"]
    
    print(f"🏋️ Training multi-modal model from epoch {start_epoch} on {config['device']}...")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config["num_epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for video, input_annotations, output_annotations in train_pbar:
            video = video.to(config["device"])
            input_annotations = input_annotations.to(config["device"])
            output_annotations = output_annotations.to(config["device"])
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                outputs = model(video, input_annotations)
                loss = criterion(outputs, output_annotations.to(outputs.dtype))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for video, input_annotations, output_annotations in val_loader:
                video = video.to(config["device"])
                input_annotations = input_annotations.to(config["device"])
                output_annotations = output_annotations.to(config["device"])
                
                with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                    outputs = model(video, input_annotations)
                    loss = criterion(outputs, output_annotations.to(outputs.dtype))
                
                val_loss += loss.item()
                # Use optimized threshold for positive class detection
                preds = (torch.sigmoid(outputs.float()) > 0.4).float()
                all_val_preds.append(preds.cpu().numpy())
                all_val_targets.append(output_annotations.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate comprehensive multi-modal metrics
        val_preds = np.vstack(all_val_preds)
        val_targets = np.vstack(all_val_targets)
        
        # 1. Overall accuracy (exact match for all outputs) - all labels must be correct
        exact_match_acc = accuracy_score(val_targets, val_preds)
        
        # 2. Simple accuracy - correct_predictions / total_predictions across all outputs
        simple_acc = np.mean(val_preds == val_targets)
        
        # 3. Positive Class F1 Scores (optimized for detecting presence)
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(val_targets, val_preds, average='micro', zero_division=0)
        
        # Additional positive class metrics
        pos_rate_target = val_targets.mean()
        pos_rate_pred = val_preds.mean()
        macro_precision = precision_score(val_targets, val_preds, average='macro', zero_division=0)
        macro_recall = recall_score(val_targets, val_preds, average='macro', zero_division=0)
        
        # 5. Per-output detailed metrics
        per_output_accs = []
        per_output_f1s = []
        per_output_precision = []
        per_output_recall = []
        
        for i in range(len(output_markers)):
            if val_targets[:, i].sum() > 0:  # Only if positive samples exist
                acc = accuracy_score(val_targets[:, i], val_preds[:, i])
                f1 = f1_score(val_targets[:, i], val_preds[:, i], zero_division=0)
                precision = precision_score(val_targets[:, i], val_preds[:, i], zero_division=0)
                recall = recall_score(val_targets[:, i], val_preds[:, i], zero_division=0)
                
                per_output_accs.append(acc)
                per_output_f1s.append(f1)
                per_output_precision.append(precision)
                per_output_recall.append(recall)
        
        avg_output_acc = np.mean(per_output_accs) if per_output_accs else 0.0
        avg_output_f1 = np.mean(per_output_f1s) if per_output_f1s else 0.0
        avg_output_precision = np.mean(per_output_precision) if per_output_precision else 0.0
        avg_output_recall = np.mean(per_output_recall) if per_output_recall else 0.0
        
        print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        print(f"    Exact Match Acc={exact_match_acc:.4f} | Simple Acc={simple_acc:.4f} | Avg Output Acc={avg_output_acc:.4f}")
        print(f"    🎯 POSITIVE CLASS: F1={macro_f1:.4f} | Precision={macro_precision:.4f} | Recall={macro_recall:.4f}")
        print(f"    🎯 Positive Rate: Target={pos_rate_target:.3f} | Pred={pos_rate_pred:.3f} | Micro F1={micro_f1:.4f}")
        
        # Logging
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'exact_match_accuracy': exact_match_acc,
            'simple_accuracy': simple_acc,
            'avg_output_accuracy': avg_output_acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'avg_output_f1': avg_output_f1,
            'avg_output_precision': avg_output_precision,
            'avg_output_recall': avg_output_recall
        }
        
        # Add per-output metrics
        for i, marker in enumerate(output_markers):
            if i < len(per_output_accs):
                epoch_data[f'{marker.lower().replace(" ", "_")}_accuracy'] = per_output_accs[i]
                epoch_data[f'{marker.lower().replace(" ", "_")}_f1'] = per_output_f1s[i]
                epoch_data[f'{marker.lower().replace(" ", "_")}_precision'] = per_output_precision[i]
                epoch_data[f'{marker.lower().replace(" ", "_")}_recall'] = per_output_recall[i]
        
        log_epoch(config["log_dir"], epoch_data)
        
        # Use macro F1 for checkpointing (optimized for positive class detection)
        current_val_acc = macro_f1
        
        # Save checkpoint and best model
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
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
                'exact_match_accuracy': exact_match_acc,
                'avg_output_f1': avg_output_f1
            }
            save_checkpoint(checkpoint_state, config["checkpoint_file"])
            
            # Save best model
            os.makedirs(config["model_dir"], exist_ok=True)
            model_path = os.path.join(config["model_dir"], "best_multi_modal_model.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"  🛑 Early stopping triggered")
                break
    
    return {
        "status": "completed",
        "best_avg_output_accuracy": best_val_acc,
        "final_exact_match_accuracy": exact_match_acc,
        "final_avg_output_f1": avg_output_f1,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "epochs_trained": epoch + 1,
        "input_markers": input_markers,
        "output_markers": output_markers,
        "per_output_final_accuracies": per_output_accs,
        "per_output_final_f1s": per_output_f1s
    }

# --- MAIN FUNCTION ---

def main():
    """Main execution function for multi-modal training"""
    video_locations = [
        "/home/tjeei/MultiModal Project/Big Interview/Just Face",
        "/home/tjeei/MultiModal Project/Genex/Just Respondent"
    ]
    
    print(f"✅ Domain mapping configuration loaded (14 emotions → 33 behaviors).")
    
    # Load markers list
    markers_file = os.path.join(CONFIG["data_dir"], "markers_47.txt")
    with open(markers_file, "r") as f:
        all_markers = [line.strip() for line in f if line.strip()]
    
    # Load or create preprocessed data
    if CONFIG["use_optimized_preprocessing"]:
        print("🚀 Using optimized preprocessing approach...")
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
        
        # Create optimized dataset
        if preprocessed_files:
            print("⚡ Using OptimizedMultiModalDataset")
            dataset = OptimizedMultiModalDataset(
                preprocessed_files, 
                DOMAIN_MAPPING_CONFIG["emotion_to_behavior"]["input_markers"],
                DOMAIN_MAPPING_CONFIG["emotion_to_behavior"]["output_markers"], 
                all_markers
            )
            
            # Check for sufficient data from the optimized dataset
            if len(dataset) == 0:
                print("⚠️ No data available - check marker coverage in preprocessed files")
                return
                
            print(f"📊 Dataset created with {len(dataset)} samples")
        else:
            print("📼 Fallback to legacy approach - no preprocessed files found")
            cached_data = preprocess_and_cache_data(CONFIG, video_locations, DOMAIN_MAPPING_CONFIG["emotion_to_behavior"])
            dataset = MultiModalDataset(cached_data["frame_paths"], cached_data["input_annotations"], cached_data["output_annotations"])
    else:
        print("📼 Using legacy preprocessing approach...")
        cached_data = preprocess_and_cache_data(CONFIG, video_locations, DOMAIN_MAPPING_CONFIG["emotion_to_behavior"])
        dataset = MultiModalDataset(cached_data["frame_paths"], cached_data["input_annotations"], cached_data["output_annotations"])
    
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
    num_input_markers = DOMAIN_MAPPING_CONFIG["emotion_to_behavior"]["input_dim"]
    num_output_markers = DOMAIN_MAPPING_CONFIG["emotion_to_behavior"]["output_dim"]
    model = MultiModalNet(CONFIG["img_size"], num_input_markers, num_output_markers).to(CONFIG["device"])
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    
    # Train model
    start_time = time.time()
    results = train_multi_modal_model(model, train_loader, val_loader, CONFIG, DOMAIN_MAPPING_CONFIG["emotion_to_behavior"])
    training_time = (time.time() - start_time) / 3600
    
    results["training_time_hours"] = training_time
    results["total_parameters"] = total_params
    # Convert device to string for JSON serialization
    config_for_json = CONFIG.copy()
    config_for_json["device"] = str(CONFIG["device"])
    results["config"] = config_for_json
    results["domain_mapping_config"] = DOMAIN_MAPPING_CONFIG["emotion_to_behavior"]
    
    # Save final results
    os.makedirs(os.path.dirname(CONFIG["results_file"]), exist_ok=True)
    with open(CONFIG["results_file"], "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\\n🎉 EXPERIMENT 4 COMPLETE!")
    print(f"⏱️ Training time: {training_time:.2f} hours")
    print(f"🎯 Best validation accuracy: {results.get('best_val_accuracy', 'N/A')}")
    print(f"🧠 Emotions → Behaviors mapping trained")
    print(f"📁 Results saved to: {CONFIG['results_file']}")
    print(f"📁 Model saved to: {CONFIG['model_dir']}")

if __name__ == "__main__":
    main()