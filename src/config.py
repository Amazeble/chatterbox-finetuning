from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union
import os
import glob
import json

# Path to a temporary override file created by Colab notebooks
_OVERRIDE_FILE = "colab_config_override.json"

def _get_field_value(key: str, default):
    """Helper to get value from override file or return default."""
    if os.path.exists(_OVERRIDE_FILE):
        try:
            with open(_OVERRIDE_FILE, 'r') as f:
                overrides = json.load(f)
                if key in overrides and overrides[key] is not None:
                    return overrides[key]
        except Exception:
            pass
    # Return default if no override found
    return default


def _require_field_value(key: str):
    """Helper to get value from override file, raise error if not found."""
    if os.path.exists(_OVERRIDE_FILE):
        try:
            with open(_OVERRIDE_FILE, 'r') as f:
                overrides = json.load(f)
                if key in overrides and overrides[key] is not None:
                    return overrides[key]
        except Exception:
            pass
    raise ValueError(
        f"{key} is not set. Please run the Colab configuration cell first "
        f"to set {key}, or ensure colab_config_override.json exists with a valid value."
    )


def should_run_preprocessing(config) -> bool:
    """
    Determine if preprocessing should run based on config.preprocess setting.
    
    - If preprocess is True: always run
    - If preprocess is False: skip
    - If preprocess is "auto": check if preprocessed_dir exists and has same number of .pt files as .wav files
    
    Returns True if preprocessing should run, False otherwise.
    """
    if config.preprocess is True:
        return True
    elif config.preprocess is False:
        return False
    elif config.preprocess == "auto":
        # Check if preprocessed_dir exists
        if not os.path.exists(config.preprocessed_dir):
            return True
        
        # Count .wav files in wav_dir
        wav_files = glob.glob(os.path.join(config.wav_dir, "*.wav"))
        wav_count = len(wav_files)
        
        # Count .pt files in preprocessed_dir
        pt_files = glob.glob(os.path.join(config.preprocessed_dir, "*.pt"))
        pt_count = len(pt_files)
        
        # If counts don't match, rerun preprocessing
        if wav_count != pt_count:
            return True
        
        return False
    
    # Default to running preprocessing
    return True


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
    # Value MUST be set via colab_config_override.json (created by running the Colab config cell)
    # If empty, paths will use base_dataset_dir directly without subfolder
    project_name: str = field(default_factory=lambda: _get_field_value("project_name", ""))
    
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
    # Preprocessing mode: True (always run), False (skip), "auto" (smart detection)
    preprocess: Optional[Union[bool, Literal["auto"]]] = field(default_factory=lambda: _get_field_value("preprocess", True))
    
    is_turbo: bool = field(default_factory=lambda: _get_field_value("is_turbo", True))  # Set True if you're training Turbo, False if you're training Normal.
    is_lora: bool = field(default_factory=lambda: _get_field_value("is_lora", True))   # True: Efficient LoRA training (Recommended for < 10h data)
                           # False: Full Fine-Tune (High VRAM, for massive datasets)
    is_merge_lora: bool = field(default_factory=lambda: _get_field_value("is_merge_lora", False))  # If True and is_lora is True, automatically run merge_lora.py after training

    lora_r: int = field(default_factory=lambda: _get_field_value("lora_r", 128))
    lora_alpha: int = field(default_factory=lambda: _get_field_value("lora_alpha", 256))
    turbo_lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc", "spkr_enc"])
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "spkr_enc"])
    lora_modules_to_save: List[str] = field(default_factory=lambda: ["text_emb", "text_head"])
    

    # --- Vocabulary ---
    # The size of the NEW vocabulary (from tokenizer.json)
    # Ensure this matches the JSON file generated by your tokenizer script.
    # For Turbo mode: Use the exact number provided by setup.py (e.g., 52260)
    new_vocab_size: int = field(default_factory=lambda: _get_field_value("new_vocab_size", 52260 if _get_field_value("is_turbo", True) else 2454))

    # --- Hyperparameters ---
    batch_size: int = field(default_factory=lambda: _get_field_value("batch_size", 32))      # Adjust based on VRAM (2, 4, 8)
    grad_accum: int = field(default_factory=lambda: _get_field_value("grad_accum", 4))       # Effective Batch Size = Batch * Accum
    learning_rate: float = field(default_factory=lambda: _get_field_value("learning_rate", 0.0001 if _get_field_value("is_lora", True) else 0.00001))  # T3 is sensitive, keep low
    num_epochs: int = field(default_factory=lambda: _get_field_value("num_epochs", 10 if _get_field_value("is_lora", True) else 30))
    
    save_steps: int = field(default_factory=lambda: _get_field_value("save_steps", 500))
    save_total_limit: int = field(default_factory=lambda: _get_field_value("save_total_limit", 2))
    dataloader_num_workers: int = field(default_factory=lambda: _get_field_value("dataloader_num_workers", 2))

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850   # Truncates very long audio
    prompt_duration: float = 3.0 # Duration for the reference prompt (seconds)
    
    def __post_init__(self):
        """Resolve paths using project_name"""
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
