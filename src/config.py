from dataclasses import dataclass, field
from typing import List, Optional
import os
import glob
import json
import hashlib


def find_project_name(base_dataset_dir: str) -> str:
    """
    Automatically detect project name by finding subdirectories in base_dataset_dir.
    Looks for directories under ./TTSDataset/** or the configured base_dataset_dir.
    
    Returns the first found subdirectory name, or empty string if none found.
    """
    # Check if base_dataset_dir exists
    if not os.path.exists(base_dataset_dir):
        return ""
    
    # Find all subdirectories in base_dataset_dir
    subdirs = [d for d in os.listdir(base_dataset_dir) 
               if os.path.isdir(os.path.join(base_dataset_dir, d))]
    
    # Filter for directories that look like project folders (have wavs or metadata.csv)
    for subdir in subdirs:
        subdir_path = os.path.join(base_dataset_dir, subdir)
        # Check if this looks like a valid project directory
        if (os.path.exists(os.path.join(subdir_path, "wavs")) or 
            os.path.exists(os.path.join(subdir_path, "metadata.csv"))):
            return subdir
    
    # If no valid project directory found, return first subdirectory if any
    if subdirs:
        return subdirs[0]
    
    return ""


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def should_run_preprocessing(config) -> bool:
    """
    Determine if preprocessing should run using hash-based verification.
    
    Checks if preprocessed_dir exists and has a valid preprocess_report.json.
    Compares current wav file hashes with stored hashes to detect changes.
    
    Returns True if preprocessing should run, False otherwise.
    """
    # Check if preprocessed_dir exists
    if not os.path.exists(config.preprocessed_dir):
        return True
    
    # Check if preprocess_report.json exists
    report_path = os.path.join(config.preprocessed_dir, "preprocess_report.json")
    if not os.path.exists(report_path):
        return True
    
    # Load the stored report
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError):
        return True
    
    # Count source files based on dataset type
    if config.json_format:
        # For JSON format, check if metadata file exists
        if not os.path.exists(config.metadata_path):
            return True
        # We can't easily verify without re-reading, so trust the report exists
        # The first_file hash will be checked below
        wav_files = []
        wav_count = report.get("total_files", 0)
    elif config.ljspeech:
        # For LJSpeech format, count entries in CSV
        if not os.path.exists(config.csv_path):
            return True
        import pandas as pd
        try:
            data = pd.read_csv(config.csv_path, sep="|", header=None, quoting=3)
            wav_count = len(data)
        except Exception:
            return True
        wav_files = []
    else:
        # For file-based format, count .wav files in wav_dir
        wav_files = sorted(glob.glob(os.path.join(config.wav_dir, "*.wav")))
        wav_count = len(wav_files)
    
    # Check if total file count matches
    if wav_count != report.get("total_files", 0):
        return True
    
    # If no files, no need to preprocess
    if wav_count == 0:
        return False
    
    # Compare first file hash for verification
    # Get the first filename from the report
    stored_first_filename = report.get("first_file", {}).get("filename", "")
    
    if not stored_first_filename:
        return True
    
    # Construct the path to the first wav file
    first_wav = os.path.join(config.wav_dir, stored_first_filename)
    
    if not os.path.exists(first_wav):
        return True
    
    # Compare first file hash
    current_first_hash = compute_file_hash(first_wav)
    stored_first_hash = report.get("first_file", {}).get("hash", "")
    
    if current_first_hash != stored_first_hash:
        return True
    
    return False


@dataclass
class TrainConfig:
    # --- Paths ---
    # Directory where setup.py downloaded the files
    model_dir: str = "./pretrained_models"
    
    # Base dataset directory - will be combined with project_name if project_name is not empty
    base_dataset_dir: str = "/content/drive/MyDrive/Chatterbox/MyTTSDataset"
    
    # Base output directory - will be combined with project_name
    base_output_dir: str = "/content/drive/MyDrive/Chatterbox/chatterbox_output"
    
    # Project name for organizing dataset and outputs (e.g., "Adriene")
    # If empty, paths will use base_dataset_dir directly without subfolder
    # Automatically detected from ./TTSDataset/** if not provided
    project_name: Optional[str] = None  # Set dynamically in __post_init__
    
    # Path to your metadata CSV (Format: ID|RawText|NormText)
    # Will be resolved as {base_dataset_dir}/{project_name}/metadata.csv if project_name is not empty
    # Otherwise resolved as {base_dataset_dir}/metadata.csv
    csv_path: str = None  # Set dynamically in __post_init__
    
    # Directory containing WAV files
    # Will be resolved as {base_dataset_dir}/{project_name}/wavs if project_name is not empty
    # Otherwise resolved as {base_dataset_dir}/wavs
    wav_dir: str = None  # Set dynamically in __post_init__
    
    preprocessed_dir: str = None  # Set dynamically in __post_init__
    
    # Output directory for the finetuned model (resolved dynamically)
    output_dir: str = None  # Set dynamically in __post_init__
    
    is_inference = False
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı ve şimdi sahip olduğuma göre, sessiz kalmayacağım."


    ljspeech = True # Set True if the dataset format is ljspeech, and False if it's file-based.
    json_format = False # Set True if the dataset format is json, and False if it's file-based or ljspeech.
    
    is_turbo: bool = True  # Set True if you're training Turbo, False if you're training Normal.
    is_lora: bool = True   # True: Efficient LoRA training (Recommended for < 10h data)
                           # False: Full Fine-Tune (High VRAM, for massive datasets)
    is_merge_lora: bool = False  # If True and is_lora is True, automatically run merge_lora.py after training

    lora_r: int = 128
    lora_alpha: int = 256
    turbo_lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc", "spkr_enc"])
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "spkr_enc"])
    lora_modules_to_save: List[str] = field(default_factory=lambda: ["text_emb", "text_head"])
    

    # --- Vocabulary ---
    # The size of the NEW vocabulary (from tokenizer.json)
    # Ensure this matches the JSON file generated by your tokenizer script.
    # For Turbo mode: Use the exact number provided by setup.py (e.g., 52260)
    new_vocab_size: int = 52260

    # --- Hyperparameters ---
    batch_size: int = 32      # Adjust based on VRAM (2, 4, 8)
    grad_accum: int = 4       # Effective Batch Size = Batch * Accum
    learning_rate: float = 0.0001  # T3 is sensitive, keep low
    num_epochs: int = 10
    
    save_steps: int = 500
    save_total_limit: int = 2
    dataloader_num_workers: int = 2

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850   # Truncates very long audio
    prompt_duration: float = 3.0 # Duration for the reference prompt (seconds)
    
    def __post_init__(self):
        """Resolve paths using project_name"""
        # Auto-detect project_name if not provided
        if self.project_name is None:
            self.project_name = find_project_name(self.base_dataset_dir)
        
        # Resolve dataset paths - use subfolder only if project_name is not empty
        if self.project_name:
            self.csv_path = os.path.join(self.base_dataset_dir, self.project_name, "metadata.csv")
            self.wav_dir = os.path.join(self.base_dataset_dir, self.project_name, "wavs")
            self.preprocessed_dir = os.path.join(self.base_dataset_dir, self.project_name, "preprocess")
            self.output_dir = os.path.join(self.base_output_dir, self.project_name)
        else:
            self.csv_path = os.path.join(self.base_dataset_dir, "metadata.csv")
            self.wav_dir = os.path.join(self.base_dataset_dir, "wavs")
            self.preprocessed_dir = os.path.join(self.base_dataset_dir, "preprocess")
            self.output_dir = self.base_output_dir
