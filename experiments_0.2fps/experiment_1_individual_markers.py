#!/usr/bin/env python3
"""
Experiment 1: Individual Marker Training - CLEAN VERSION
========================================================
Features:
- Automatic Mixed Precision (AMP) for faster training
- Smart checkpoint resuming (skips completed markers)
- Every 5th frame sampling (0.2 FPS) for efficiency
- Small & fast CNN architecture (~300K parameters)
- Comprehensive logging and result tracking
- Clean subfolder organization
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

# --- Configuration ---
BASE_DIR = "/home/tjeei/MultiModal Collab"
CONFIG = {
    "fps": 0.2,  # Every 5th frame for efficiency
    "img_size": 224,
    "batch_size": 64,
    "num_epochs": 12,
    "learning_rate": 0.002,
    "train_split": 0.8,
    "num_workers": os.cpu_count(),
    "device": get_optimal_device(),
    "use_amp": True,
    "data_dir": "/home/tjeei/MultiModal Collab",  # Remote WSL data location
    "base_dir": BASE_DIR,
    "results_file": f"{BASE_DIR}/experiment_1/results/experiment_1_results.json",
    "padding_strategy": "repeat_last",
    "frame_cache_dir": f"{BASE_DIR}/experiment_1/frame_cache",
    "metadata_cache_file": f"{BASE_DIR}/experiment_1/processed_data_cache.pkl",
    "checkpoint_dir": f"{BASE_DIR}/experiment_1/checkpoints",
    "log_dir": f"{BASE_DIR}/experiment_1/logs",
    "model_dir": f"{BASE_DIR}/experiment_1/models",
    "force_recache": False,
    "early_stopping_patience": 5,
    # New preprocessing options
    "use_optimized_preprocessing": True,  # Use new one-time preprocessing
    "preprocessed_data_dir": f"{BASE_DIR}/preprocessed_data",
    "force_reprocess": False  # Force reprocessing of videos+annotations
}

print("🚀 EXPERIMENT 1: INDIVIDUAL MARKER TRAINING")
print("="*70)
print(f"💻 Device: {CONFIG['device']}")
print(f"📊 Batch Size: {CONFIG['batch_size']}")
print(f"⚡ Automatic Mixed Precision: {CONFIG['use_amp']}")
print(f"🎯 FPS: {CONFIG['fps']} (every 5th frame)")
print(f"📈 Epochs: {CONFIG['num_epochs']}")
print(f"🧠 Model: SmallFastCNN (~300K parameters)")
print(f"⚙️ Dataloader Workers: {CONFIG['num_workers']}")
print(f"📁 Output Directory: {CONFIG['base_dir']}/experiment_1/")
print("="*70)

# --- DATASET & MODEL ---

class LazyFrameDataset(Dataset):
    def __init__(self, frame_paths, labels, preprocessed_data=None):
        self.frame_paths = frame_paths
        self.labels = labels
        self.preprocessed_data = preprocessed_data
        self._cached_data = {}
        
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        label = self.labels[idx]
        
        # Check if this is a preprocessed .npz file with participant data
        if self.preprocessed_data and frame_path.endswith('.npz') and 'preprocessed_data' in frame_path:
            # Extract participant ID and frame index from the path
            participant_file = frame_path
            
            # Cache the participant data to avoid repeated loading
            if participant_file not in self._cached_data:
                data = np.load(participant_file)
                self._cached_data[participant_file] = {
                    'frames': data['frames'],
                    'annotations': data['annotations']
                }
            
            # Get the frame index (we need to map global idx to participant frame idx)
            cached = self._cached_data[participant_file]
            frame_idx = idx % len(cached['frames'])  # Simple mapping for now
            frame = cached['frames'][frame_idx]
            
        elif frame_path.endswith('.npz'):
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
        
        return torch.tensor(frame, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class PreprocessedDataset(Dataset):
    """Optimized dataset for preprocessed participant .npz files."""
    def __init__(self, preprocessed_files, marker_name, all_markers):
        self.data = []
        self.marker_idx = all_markers.index(marker_name)
        
        # Load all data and create flat frame/label pairs
        for participant_file in preprocessed_files:
            data = np.load(participant_file)
            frames = data['frames']
            annotations = data['annotations']
            
            # Extract labels for this specific marker
            marker_labels = annotations[:, self.marker_idx]
            
            # Add each frame-label pair
            for frame_idx in range(len(frames)):
                self.data.append({
                    'frame': frames[frame_idx],
                    'label': marker_labels[frame_idx],
                    'participant_file': participant_file,
                    'frame_idx': frame_idx
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frame = item['frame'].astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        label = item['label']
        
        return torch.tensor(frame, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class SmallFastCNN(nn.Module):
    """Small & Fast CNN for efficient training with ~300K parameters"""
    def __init__(self, img_size):
        super(SmallFastCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        conv_output_size = (img_size // 8) ** 2 * 48
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze()

# --- DATA PROCESSING FUNCTIONS ---

def process_video_with_caching(video_path, frame_cache_dir, fps, img_size):
    """Process video frames with caching for faster subsequent runs."""
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

def preprocess_all_data_once(config, video_locations, all_markers):
    """One-time preprocessing that saves all frames + annotations for all experiments."""
    preprocessed_data_dir = os.path.join(config["base_dir"], "preprocessed_data")
    
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
    print(f"📊 Total participants: {len(processed_data)}")
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
                "num_frames": len(data['frames']),
                "frame_paths": [file_path] * len(data['frames'])  # For compatibility
            })
    
    # Create compatibility structure
    max_frames = max(item["num_frames"] for item in processed_data) if processed_data else 0
    
    # Add padding info for compatibility
    for item in processed_data:
        padding_needed = max_frames - item["num_frames"]
        if padding_needed > 0:
            item["frame_paths"].extend([item["file_path"]] * padding_needed)
    
    return {"max_frames": max_frames, "data": processed_data}

def preprocess_and_cache_data(config, video_locations, all_markers):
    """Preprocess and cache video data with proper frame alignment.""" 
    # First try the new optimized preprocessing
    return preprocess_all_data_once(config, video_locations, all_markers)
    
    # Legacy fallback (keeping for compatibility)
    metadata_cache_file = config["metadata_cache_file"]
    
    if os.path.exists(metadata_cache_file) and not config["force_recache"]:
        print(f"⚡️ Loading pre-processed data from {metadata_cache_file}")
        with open(metadata_cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"🔄 Pre-processing videos (FPS: {config['fps']}, every {int(1/config['fps'])}th frame)...")
    
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
            
            # Create frame-level labels
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
    
    # Save cache
    os.makedirs(os.path.dirname(metadata_cache_file), exist_ok=True)
    cache_payload = {"max_frames": max_frames, "data": processed_data}
    with open(metadata_cache_file, 'wb') as f:
        pickle.dump(cache_payload, f)
    
    print(f"✅ Pre-processing complete. Data cached to {metadata_cache_file}")
    return cache_payload

# --- CHECKPOINT & LOGGING FUNCTIONS ---

def get_completed_markers(checkpoint_dir):
    """Get list of markers that already have completed training."""
    if not os.path.exists(checkpoint_dir):
        return set()
    
    completed = set()
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("ckpt_") and filename.endswith(".pth.tar"):
            marker = filename.replace("ckpt_", "").replace(".pth.tar", "").replace("_", " ")
            completed.add(marker)
    
    return completed

def save_checkpoint(state, marker_name, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f"ckpt_{marker_name.replace(' ', '_')}.pth.tar")
    torch.save(state, filename)

def load_checkpoint(model, optimizer, scaler, marker_name, checkpoint_dir, config):
    filename = os.path.join(checkpoint_dir, f"ckpt_{marker_name.replace(' ', '_')}.pth.tar")
    if not os.path.exists(filename):
        return 0, 0.0
    
    print(f" 📂 Resuming from checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location=config["device"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if config["use_amp"] and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_accuracy']
    return start_epoch, best_val_acc

def log_epoch(log_dir, marker_name, epoch_data):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{marker_name.replace(' ', '_')}.csv")
    log_df = pd.DataFrame([epoch_data])
    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)

def save_individual_model_outputs(model, val_loader, config, marker_name):
    """Save model predictions, logits, and ground truth for individual marker analysis"""
    model.eval()
    
    all_logits = []
    all_predictions = []
    all_targets = []
    all_frame_ids = []
    
    print(f"💾 Saving outputs for {marker_name}...")
    
    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(val_loader):
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
    all_logits = np.concatenate(all_logits)
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Save outputs
    outputs_dir = os.path.join(config["base_dir"], "model_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    safe_marker_name = marker_name.replace(' ', '_').replace('/', '_')
    output_file = os.path.join(outputs_dir, f"experiment_1_outputs_{safe_marker_name}.npz")
    np.savez_compressed(
        output_file,
        logits=all_logits,
        predictions=all_predictions,
        targets=all_targets,
        frame_ids=all_frame_ids,
        marker_name=marker_name
    )
    
    print(f"✅ {marker_name} outputs saved to {output_file}")
    print(f"   Shape: {all_logits.shape} (samples,)")
    
    return output_file

# --- TRAINING FUNCTION ---

def train_marker_model(model, train_loader, val_loader, marker_name, config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, scaler, marker_name, config["checkpoint_dir"], config)
    
    print(f"🏋️ Training {marker_name} from epoch {start_epoch} on {config['device']}...")
    
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
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(config["device"]), labels.to(config["device"])
                with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                    outputs = model(frames)
                    loss = criterion(outputs, labels.to(outputs.dtype))
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs.float()) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate comprehensive metrics for binary classification
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_precision = precision_score(val_targets, val_preds, zero_division=0)
        val_recall = recall_score(val_targets, val_preds, zero_division=0)
        
        print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        print(f"    Acc={val_acc:.4f} | F1={val_f1:.4f} | Precision={val_precision:.4f} | Recall={val_recall:.4f}")
        
        # Logging
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall
        }
        log_epoch(config["log_dir"], marker_name, epoch_data)
        
        # Save checkpoint and best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall
            }
            save_checkpoint(checkpoint_state, marker_name, config["checkpoint_dir"])
            
            # Save best model
            os.makedirs(config["model_dir"], exist_ok=True)
            model_path = os.path.join(config["model_dir"], f"best_model_{marker_name.replace(' ', '_')}.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"  🛑 Early stopping triggered for {marker_name}")
                break
    
    # Save model outputs for further analysis
    print(f"\n💾 Saving best model outputs for {marker_name}...")
    output_file = save_individual_model_outputs(model, val_loader, config, marker_name)
    
    # Final metrics to return
    final_f1 = val_f1
    final_precision = val_precision
    final_recall = val_recall
    
    return {
        "status": "completed",
        "best_val_accuracy": best_val_acc,
        "final_f1": final_f1,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "epochs_trained": epoch + 1,
        "output_file": output_file
    }

# --- MAIN FUNCTION ---

def main():
    """Main execution function with smart checkpoint resuming."""
    video_locations = [
        "/home/tjeei/MultiModal Collab/Big Interview/Just Face",
        "/home/tjeei/MultiModal Collab/Genex/Just Respondent"
    ]
    
    # Load markers
    markers_file = os.path.join("markers_47.txt")
    with open(markers_file, "r") as f:
        all_markers = [line.strip() for line in f if line.strip()]
    
    print(f"✅ {len(all_markers)} markers loaded.")
    
    # Check completed markers
    completed_markers = get_completed_markers(CONFIG["checkpoint_dir"])
    remaining_markers = [m for m in all_markers if m not in completed_markers]
    
    print(f"✅ {len(completed_markers)} markers already completed.")
    print(f"⏳ {len(remaining_markers)} markers remaining to train.")
    
    if not remaining_markers:
        print("🎉 All markers already completed!")
        return
    
    print(f"🎯 Will train: {remaining_markers[:5]}{'...' if len(remaining_markers) > 5 else ''}")
    
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
    else:
        # Fallback to legacy approach
        print("📼 Using legacy preprocessing approach...")
        cached_data = preprocess_and_cache_data(CONFIG, video_locations, all_markers)
        preprocessed_files = []
    
    all_video_data = cached_data["data"]
    
    # Train remaining markers
    results = {}
    total_start_time = time.time()
    
    for i, marker in enumerate(remaining_markers):
        print("\\n" + "="*70)
        print(f"🎯 PROCESSING MARKER {i+1}/{len(remaining_markers)}: {marker}")
        print("="*70)
        
        # Use optimized dataset if available
        if CONFIG["use_optimized_preprocessing"] and preprocessed_files:
            print(f"⚡ Using optimized PreprocessedDataset for {marker}")
            full_dataset = PreprocessedDataset(preprocessed_files, marker, all_markers)
            all_frames_paths, all_labels_for_marker = [], []  # Not needed for new dataset
        else:
            # Fallback to legacy approach
            print(f"📼 Using legacy dataset approach for {marker}")
            all_frames_paths, all_labels_for_marker = [], []
            for video_data in all_video_data:
                all_frames_paths.extend(video_data["frame_paths"])
                all_labels_for_marker.extend(video_data["labels"][marker])
            full_dataset = None
        
        # Check for sufficient data and create datasets
        if CONFIG["use_optimized_preprocessing"] and full_dataset is not None:
            # Using optimized preprocessing
            dataset = full_dataset
            
            # Calculate positive samples from the dataset
            positive_samples = sum(1 for i in range(len(dataset)) if dataset.data[i]['label'] > 0.5)
            total_frames = len(dataset)
            
            if positive_samples < 10:
                print(f"⚠️ Skipping {marker} - insufficient positive samples ({int(positive_samples)}).")
                results[marker] = {"status": "skipped", "reason": "insufficient data"}
                continue
            
            print(f"📊 Data: {total_frames} frames, {int(positive_samples)} positive samples (optimized)")
        else:
            # Using legacy approach
            positive_samples = sum(all_labels_for_marker)
            if positive_samples < 10:
                print(f"⚠️ Skipping {marker} - insufficient positive samples ({int(positive_samples)}).")
                results[marker] = {"status": "skipped", "reason": "insufficient data"}
                continue
            
            print(f"📊 Data: {len(all_frames_paths)} frames, {int(positive_samples)} positive samples (legacy)")
            dataset = LazyFrameDataset(all_frames_paths, all_labels_for_marker)
        
        # Create train/val split
        train_size = int(CONFIG["train_split"] * len(dataset))
        val_size = len(dataset) - train_size
        g = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                                shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], 
                              shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        
        # Train model
        model = SmallFastCNN(CONFIG["img_size"]).to(CONFIG["device"])
        training_results = train_marker_model(model, train_loader, val_loader, marker, CONFIG)
        results[marker] = training_results
        
        # Progress update
        elapsed_total = time.time() - total_start_time
        avg_time_per_marker = elapsed_total / (i + 1)
        eta_seconds = (len(remaining_markers) - (i + 1)) * avg_time_per_marker
        
        print(f"📊 Progress: {i+1}/{len(remaining_markers)} | Elapsed: {elapsed_total/3600:.2f}h | ETA: {eta_seconds/3600:.2f}h")
    
    # Save final results
    os.makedirs(os.path.dirname(CONFIG["results_file"]), exist_ok=True)
    with open(CONFIG["results_file"], "w") as f:
        json.dump(results, f, indent=4)
    
    total_time = (time.time() - total_start_time) / 3600
    print(f"\\n🎉 EXPERIMENT 1 COMPLETE!")
    print(f"⏱️ Total time: {total_time:.2f} hours")
    print(f"📁 Results saved to: {CONFIG['results_file']}")
    print(f"📁 Models saved to: {CONFIG['model_dir']}")
    print(f"📁 Logs saved to: {CONFIG['log_dir']}")

if __name__ == "__main__":
    main()