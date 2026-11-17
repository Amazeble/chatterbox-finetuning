import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    # --- Model selection ---
    is_turbo: bool = True  # True: Turbo (GPT-2 medium). False: Standard.
    is_nano: bool = True   # True: Nano (GPT-2 small). Requires is_turbo=True.
    
    extend_vocab: bool = False  # True: Extend the tokenizer vocab. False: Use the base model vocab.
    
    # Separate folders per variant (the merge overwrites tokenizer files in model_dir)
    _sfx = ("_nano" if extend_vocab else "_nano_bpe") if is_nano else ("_turbo" if extend_vocab else "_turbo_bpe") if is_turbo else ""

    # --- Training strategy ---
    is_lora: bool = True  # True: LoRA (recommended for < 10h data). False: full fine-tune.
    
    # --- Paths ---
    # Optional project isolation. Set project_name (e.g. "Elise") to enable it.
    # By default, files are placed under ./project/<project_name>.
    # Set project_base_dir to override that directory explicitly.
    project_name: str | None = "Elise"
    project_base_dir: str | None = None

    # Directory where setup.py downloads the base model files
    model_dir: str = f"./pretrained_models{_sfx}"
    
    # Path to your metadata CSV (Format: ID|RawText|NormText)
    csv_path: str = "./MyTTSDataset/metadata.csv"
    #metadata_path: str = "./MyTTSDataset/metadata_final.json"
    
    # Directory containing WAV files
    wav_dir: str = "./MyTTSDataset/wavs"
    #wav_dir: str = "./FileBasedDataset"
    
    preprocessed_dir: str = f"./preprocess{_sfx}"
    #preprocessed_dir = "./FileBasedDataset/preprocess{_sfx}"
    
    # Output directory for the finetuned model
    output_dir: str = f"./chatterbox_output{_sfx}"
    
    is_inference = True
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı ve şimdi sahip olduğuma göre, sessiz kalmayacağım."

    # --- Dataset ---
    ljspeech = True      # True: LJSpeech format (metadata.csv)
    json_format = False  # True: JSON format. False (with ljspeech=False): file-based.
    preprocess = False    # Set False if the dataset is already preprocessed.

    lora_r: int = 32
    lora_alpha: int = 64
    turbo_lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc", "spkr_enc"])
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "spkr_enc"])
    lora_modules_to_save: List[str] = field(default_factory=lambda: ["text_emb", "text_head"])

    # --- Vocabulary ---
    # Must equal len(tokenizer). Turbo/Nano: use the value printed by setup.py.
    new_vocab_size: int = (52274 if extend_vocab else 50276) if is_turbo else 2454

    # --- Hyperparameters ---
    batch_size: int = 8   # Adjust to VRAM
    grad_accum: int = 2    # Effective batch = batch_size * grad_accum
    learning_rate: float = 1e-4 if is_lora else 1e-5
    num_epochs: int = 14 if is_lora else 20
    save_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 2
    dataloader_num_workers: int = 4

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850     # Longer audio is truncated
    prompt_duration: float = 3.0  # Reference prompt length (seconds)

    def __post_init__(self):
        assert not (self.is_nano and not self.is_turbo), "is_nano=True requires is_turbo=True"

        # Keep the original paths when project_name is empty/None.
        # When enabled, prefix only the dataset/preprocessing/output paths.
        if self.project_name and self.project_name.strip():
            if not self.project_base_dir:
                self.project_base_dir = os.path.join("./project", self.project_name.strip())

            self.project_base_dir = os.path.normpath(self.project_base_dir)
            if not os.path.isabs(self.project_base_dir):
                self.project_base_dir = os.path.join(".", self.project_base_dir)

            for path_attr in ("csv_path", "wav_dir", "preprocessed_dir", "output_dir"):
                resolved_path = os.path.normpath(
                    os.path.join(self.project_base_dir, getattr(self, path_attr))
                )
                if not os.path.isabs(resolved_path):
                    resolved_path = os.path.join(".", resolved_path)
                setattr(self, path_attr, resolved_path)

    @property
    def full_ft_filename(self):
        return ("t3_nano_finetuned.safetensors" if self.is_nano else
                "t3_turbo_finetuned.safetensors" if self.is_turbo else
                "t3_finetuned.safetensors")

    @property
    def merged_filename(self):
        return self.full_ft_filename.replace("finetuned", "finetuned_merged")
