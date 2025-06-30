#!/usr/bin/env python3
"""
Experiment 2: Multi-Label Classification - STANDARDIZED VERSION
==============================================================
Extension of Experiment 1 - Single model predicting ALL markers simultaneously
Architecture: SmallFastCNN + Multiple output heads (one per marker)
Training: Multi-label BCE loss with shared feature learning

Key Standardizations from Experiment 1:
- Every 5th frame sampling (0.2 FPS) for efficiency
- Same preprocessing pipeline and data loading
- Same SmallFastCNN base architecture
- Same training protocols and optimization
- Clean subfolder organization

Goal: Demonstrate joint learning vs individual models
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, multilabel_confusion_matrix
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
    "batch_size": 32,  # Smaller batch for multi-label memory efficiency
    "num_epochs": 20,  # More epochs for multi-label convergence
    "learning_rate": 0.001,  # Lower LR for multi-label stability
    "train_split": 0.8,
    "num_workers": os.cpu_count(),
    "device": get_optimal_device(),
    "use_amp": True,
    "data_dir": "/home/tjeei/MultiModal Project",  # Remote WSL data location
    "base_dir": BASE_DIR,
    "results_file": f"{BASE_DIR}/experiment_2_1fps/results/experiment_2_1fps_results.json",
    "padding_strategy": "repeat_last",
    "frame_cache_dir": f"{BASE_DIR}/experiment_2_1fps/frame_cache",
    "metadata_cache_file": f"{BASE_DIR}/experiment_2_1fps/processed_data_cache.pkl",
    "checkpoint_file": f"{BASE_DIR}/experiment_2_1fps/checkpoints/multi_label_checkpoint.pth",
    "log_dir": f"{BASE_DIR}/experiment_2_1fps/logs",
    "model_dir": f"{BASE_DIR}/experiment_2_1fps/models",
    "force_recache": False,
    "early_stopping_patience": 5,
    # Optimized preprocessing options
    "use_optimized_preprocessing": True,
    "preprocessed_data_dir": f"{BASE_DIR}/preprocessed_data_1fps",
    "force_reprocess": False
}

print("🚀 EXPERIMENT 2: MULTI-LABEL CLASSIFICATION (1 FPS)")
print("="*70)
print(f"💻 Device: {CONFIG['device']}")
print(f"📊 Batch Size: {CONFIG['batch_size']} (multi-label optimized)")
print(f"⚡ Automatic Mixed Precision: {CONFIG['use_amp']}")
print(f"🎯 FPS: {CONFIG['fps']} (1 frame per second)")
print(f"📈 Epochs: {CONFIG['num_epochs']}")
print(f"🧠 Model: SmallFastCNN + 47 Multi-label Heads")
print(f"⚙️ Dataloader Workers: {CONFIG['num_workers']}")
print(f"📁 Output Directory: {CONFIG['base_dir']}/experiment_2_1fps/")
print("="*70)

# --- DATASET ---

class MultiLabelFrameDataset(Dataset):
    """Dataset for multi-label frame classification"""
    def __init__(self, frame_paths, multi_labels):
        self.frame_paths = frame_paths
        self.multi_labels = multi_labels  # List of label vectors (one per frame)
        
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        labels = self.multi_labels[idx]
        
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
        
        return torch.tensor(frame, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

class OptimizedMultiLabelDataset(Dataset):
    """Optimized dataset for preprocessed participant .npz files with multi-label support."""
    def __init__(self, preprocessed_files, all_markers):
        self.data = []
        
        # Load all data and create flat frame/label pairs
        for participant_file in preprocessed_files:
            data = np.load(participant_file)
            frames = data['frames']
            annotations = data['annotations']  # Shape: [num_frames, num_markers]
            
            # Add each frame with all its labels
            for frame_idx in range(len(frames)):
                self.data.append({
                    'frame': frames[frame_idx],
                    'labels': annotations[frame_idx],  # All markers for this frame
                    'participant_file': participant_file,
                    'frame_idx': frame_idx
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frame = item['frame'].astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        labels = item['labels']  # Multi-label vector
        
        return torch.tensor(frame, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Copy preprocessing functions from experiment_1
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

class MultiLabelCNN(nn.Module):
    """Multi-label CNN with SmallFastCNN backbone + multiple heads"""
    def __init__(self, img_size, num_labels):
        super(MultiLabelCNN, self).__init__()
        self.backbone = SmallFastCNN(img_size)
        self.num_labels = num_labels
        
        # Individual heads for each marker
        self.heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_labels)
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(features).squeeze())
        return torch.stack(outputs, dim=1)  # [batch_size, num_labels]

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

def preprocess_and_cache_data(config, video_locations, all_markers):
    """Preprocess and cache data for multi-label training"""
    metadata_cache_file = config["metadata_cache_file"]
    
    if os.path.exists(metadata_cache_file) and not config["force_recache"]:
        print(f"⚡️ Loading pre-processed data from {metadata_cache_file}")
        with open(metadata_cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"🔄 Pre-processing videos for multi-label training (FPS: {config['fps']})...")
    
    # Load annotation data
    big_interview_annotations = pd.read_csv(os.path.join(config["data_dir"], "Big Interview/big_interview_annotations.csv"))
    genex_annotations = pd.read_csv(os.path.join(config["data_dir"], "Genex/genex_annotations.csv"))
    
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
            
            frame_paths = process_video_with_caching(
                video_path, config["frame_cache_dir"], config["fps"], config["img_size"]
            )
            
            if not frame_paths:
                continue
            
            # Create frame-level labels for ALL markers
            labels = {marker: [0.0] * len(frame_paths) for marker in all_markers}
            
            for _, annotation in participant_annotations.iterrows():
                start_frame = int(annotation['Start Time (ms)'] / 1000 * config["fps"])
                end_frame = int(annotation['End Time (ms)'] / 1000 * config["fps"])
                marker = annotation['Marker Name']
                
                if marker in labels:
                    for frame_idx in range(max(0, start_frame), min(len(frame_paths), end_frame + 1)):
                        labels[marker][frame_idx] = 1.0
            
            processed_data.append({
                "participant_id": participant_id,
                "frame_paths": frame_paths,
                "labels": labels
            })
    
    # Padding to max frames
    max_frames = max(len(video_data["frame_paths"]) for video_data in processed_data)
    for video_data in processed_data:
        padding_needed = max_frames - len(video_data["frame_paths"])
        if padding_needed > 0:
            video_data["frame_paths"].extend([video_data["frame_paths"][-1]] * padding_needed)
            for marker in all_markers:
                video_data["labels"][marker].extend([0.0] * padding_needed)
    
    # Convert to multi-label format
    all_frame_paths = []
    all_multi_labels = []
    
    for video_data in processed_data:
        frame_paths = video_data["frame_paths"]
        labels = video_data["labels"]
        
        for i in range(len(frame_paths)):
            all_frame_paths.append(frame_paths[i])
            # Create label vector for this frame
            label_vector = [labels[marker][i] for marker in all_markers]
            all_multi_labels.append(label_vector)
    
    # Save cache
    os.makedirs(os.path.dirname(metadata_cache_file), exist_ok=True)
    cache_payload = {
        "frame_paths": all_frame_paths,
        "multi_labels": all_multi_labels,
        "markers": all_markers,
        "max_frames": max_frames
    }
    with open(metadata_cache_file, 'wb') as f:
        pickle.dump(cache_payload, f)
    
    print(f"✅ Multi-label pre-processing complete. Data cached to {metadata_cache_file}")
    print(f"📊 Total frames: {len(all_frame_paths)}, Labels per frame: {len(all_markers)}")
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
    log_file = os.path.join(log_dir, "multi_label_training.csv")
    log_df = pd.DataFrame([epoch_data])
    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)

def save_model_outputs(model, val_loader, config, all_markers):
    """Save model predictions, logits, and ground truth for analysis"""
    model.eval()
    
    all_logits = []
    all_predictions = []
    all_targets = []
    all_frame_ids = []
    
    print("💾 Saving model outputs for analysis...")
    
    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(tqdm(val_loader, desc="Saving outputs")):
            frames, labels = frames.to(config["device"]), labels.to(config["device"])
            
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                logits = model(frames)
            
            predictions = (torch.sigmoid(logits.float()) > 0.5).float()
            
            # Store batch data
            all_logits.append(logits.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
            # Create frame IDs for this batch
            batch_frame_ids = [f"batch_{batch_idx}_frame_{i}" for i in range(frames.size(0))]
            all_frame_ids.extend(batch_frame_ids)
    
    # Concatenate all data
    all_logits = np.vstack(all_logits)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Save outputs
    outputs_dir = os.path.join(config["base_dir"], "model_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    output_file = os.path.join(outputs_dir, "experiment_2_outputs.npz")
    np.savez_compressed(
        output_file,
        logits=all_logits,
        predictions=all_predictions,
        targets=all_targets,
        frame_ids=all_frame_ids,
        markers=all_markers
    )
    
    print(f"✅ Model outputs saved to {output_file}")
    print(f"   Shape: {all_logits.shape} (samples x markers)")
    
    return output_file

def train_multi_label_model(model, train_loader, val_loader, config, all_markers):
    # Calculate class weights for positive class optimization
    print("📊 Calculating class weights for positive class optimization...")
    pos_weights = []
    total_samples = 0
    total_positives = 0
    
    for frames, labels in train_loader:
        batch_pos = labels.sum(dim=0)  # Positive count per marker
        batch_total = labels.shape[0]  # Total samples in batch
        pos_weights.append(batch_pos)
        total_samples += batch_total
        total_positives += labels.sum().item()
    
    # Combine across all batches
    total_pos_per_marker = torch.stack(pos_weights).sum(dim=0)
    total_neg_per_marker = total_samples - total_pos_per_marker
    
    # Calculate positive class weights (higher weight for rarer positive class)
    pos_weight_tensor = total_neg_per_marker.float() / (total_pos_per_marker.float() + 1e-6)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=10.0)  # Reasonable bounds
    
    print(f"🎯 Positive class rate: {total_positives/(total_samples*len(all_markers)):.3f}")
    print(f"🎯 Average pos_weight: {pos_weight_tensor.mean():.2f} (range: {pos_weight_tensor.min():.2f}-{pos_weight_tensor.max():.2f})")
    
    # Use class-weighted BCE loss for positive class optimization
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(config["device"]))
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, scaler, config["checkpoint_file"], config)
    
    print(f"🏋️ Training multi-label model from epoch {start_epoch} on {config['device']}...")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config["num_epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for frames, labels in train_pbar:
            frames, labels = frames.to(config["device"]), labels.to(config["device"])
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                outputs = model(frames)
                loss = criterion(outputs, labels.to(outputs.dtype))
            
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
            for frames, labels in val_loader:
                frames, labels = frames.to(config["device"]), labels.to(config["device"])
                with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                    outputs = model(frames)
                    loss = criterion(outputs, labels.to(outputs.dtype))
                val_loss += loss.item()
                # Use optimized threshold for positive class detection (slightly lower than 0.5)
                preds = (torch.sigmoid(outputs.float()) > 0.4).float()
                all_val_preds.append(preds.cpu().numpy())
                all_val_targets.append(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate comprehensive evaluation metrics
        val_preds = np.vstack(all_val_preds)
        val_targets = np.vstack(all_val_targets)
        
        # 1. Overall accuracy (exact match) - all labels must be correct
        exact_match_acc = accuracy_score(val_targets, val_preds)
        
        # 2. Simple accuracy - correct_frames / total_frames across all labels
        simple_acc = np.mean(val_preds == val_targets)
        
        # 3. Average per-label accuracy
        per_label_accs = []
        for i in range(len(all_markers)):
            if val_targets[:, i].sum() > 0:  # Only if positive samples exist
                acc = accuracy_score(val_targets[:, i], val_preds[:, i])
                per_label_accs.append(acc)
        
        avg_label_acc = np.mean(per_label_accs) if per_label_accs else 0.0
        
        # 4. Positive Class F1 Scores (optimized for detecting presence)
        # Macro F1 (average of per-label F1 scores) - inherently positive class focused
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        
        # Micro F1 (aggregate across all labels) - inherently positive class focused  
        micro_f1 = f1_score(val_targets, val_preds, average='micro', zero_division=0)
        
        # Per-label F1 scores for positive class
        per_label_f1 = f1_score(val_targets, val_preds, average=None, zero_division=0)
        
        # Additional positive class metrics
        pos_rate_target = val_targets.mean()
        pos_rate_pred = val_preds.mean()
        macro_precision = precision_score(val_targets, val_preds, average='macro', zero_division=0)
        macro_recall = recall_score(val_targets, val_preds, average='macro', zero_division=0)
        
        print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        print(f"    Exact Match Acc={exact_match_acc:.4f} | Simple Acc={simple_acc:.4f} | Avg Label Acc={avg_label_acc:.4f}")
        print(f"    🎯 POSITIVE CLASS: F1={macro_f1:.4f} | Precision={macro_precision:.4f} | Recall={macro_recall:.4f}")
        print(f"    🎯 Positive Rate: Target={pos_rate_target:.3f} | Pred={pos_rate_pred:.3f} | Micro F1={micro_f1:.4f}")
        
        # Logging
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'exact_match_accuracy': exact_match_acc,
            'simple_accuracy': simple_acc,
            'avg_label_accuracy': avg_label_acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'positive_class_precision': macro_precision,
            'positive_class_recall': macro_recall,
            'positive_rate_target': pos_rate_target,
            'positive_rate_pred': pos_rate_pred
        }
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
                'simple_accuracy': simple_acc,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1
            }
            save_checkpoint(checkpoint_state, config["checkpoint_file"])
            
            # Save best model
            os.makedirs(config["model_dir"], exist_ok=True)
            model_path = os.path.join(config["model_dir"], "best_multi_label_model.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"  🛑 Early stopping triggered")
                break
    
    # Save model outputs for further analysis
    print(f"\n💾 Saving best model outputs...")
    output_file = save_model_outputs(model, val_loader, config, all_markers)
    
    # Final evaluation metrics to return
    final_simple_acc = simple_acc if "simple_acc" in locals() else 0.0
    final_macro_f1 = macro_f1 if "macro_f1" in locals() else 0.0
    final_micro_f1 = micro_f1 if "micro_f1" in locals() else 0.0
    final_exact_match = exact_match_acc if "exact_match_acc" in locals() else 0.0
    
    return {
        "status": "completed",
        "best_avg_label_accuracy": best_val_acc,
        "final_simple_accuracy": final_simple_acc,
        "final_macro_f1": final_macro_f1,
        "final_micro_f1": final_micro_f1,
        "final_exact_match_accuracy": final_exact_match,
        "final_train_loss": avg_train_loss if "avg_train_loss" in locals() else 0.0,
        "final_val_loss": avg_val_loss if "avg_val_loss" in locals() else 0.0,
        "epochs_trained": (epoch + 1) if "epoch" in locals() else 0,
        "total_markers": len(all_markers),
        "output_file": output_file
    }

# --- MAIN FUNCTION ---

def main():
    """Main execution function for multi-label training"""
    video_locations = [
        "/home/tjeei/MultiModal Project/Big Interview/Just Face",
        "/home/tjeei/MultiModal Project/Genex/Just Respondent"
    ]
    
    # Load markers
    markers_file = os.path.join(CONFIG["data_dir"], "markers_47.txt")
    with open(markers_file, "r") as f:
        all_markers = [line.strip() for line in f if line.strip()]
    
    print(f"✅ {len(all_markers)} markers loaded for multi-label training.")
    
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
            print("⚡ Using OptimizedMultiLabelDataset")
            dataset = OptimizedMultiLabelDataset(preprocessed_files, all_markers)
        else:
            print("📼 Fallback to legacy approach - no preprocessed files found")
            cached_data = preprocess_and_cache_data(CONFIG, video_locations, all_markers)
            dataset = MultiLabelFrameDataset(cached_data["frame_paths"], cached_data["multi_labels"])
    else:
        print("📼 Using legacy preprocessing approach...")
        cached_data = preprocess_and_cache_data(CONFIG, video_locations, all_markers)
        dataset = MultiLabelFrameDataset(cached_data["frame_paths"], cached_data["multi_labels"])
    
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
    model = MultiLabelCNN(CONFIG["img_size"], len(all_markers)).to(CONFIG["device"])
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    
    # Train model
    start_time = time.time()
    results = train_multi_label_model(model, train_loader, val_loader, CONFIG, all_markers)
    training_time = (time.time() - start_time) / 3600
    
    results["training_time_hours"] = training_time
    results["markers"] = all_markers
    
    # Save final results
    os.makedirs(os.path.dirname(CONFIG["results_file"]), exist_ok=True)
    with open(CONFIG["results_file"], "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\\n🎉 EXPERIMENT 2 COMPLETE!")
    print(f"⏱️ Training time: {training_time:.2f} hours")
    print(f"🎯 Best avg label accuracy: {results['best_avg_label_accuracy']:.4f}")
    print(f"📁 Results saved to: {CONFIG['results_file']}")
    print(f"📁 Model saved to: {CONFIG['model_dir']}")

if __name__ == "__main__":
    main()