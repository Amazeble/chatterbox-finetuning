import sys
import pkgutil
from importlib.machinery import FileFinder

# Fix 1: Handle missing ImpImporter
if not hasattr(pkgutil, 'ImpImporter'):
    class DummyImpImporter: pass
    pkgutil.ImpImporter = DummyImpImporter

# Fix 2: Handle missing find_module on FileFinder
if not hasattr(FileFinder, 'find_module'):
    def dummy_find_module(self, fullname):
        return None
    FileFinder.find_module = dummy_find_module

# Force Python to prefer pip-installed packages over broken system ones
sys.path = [p for p in sys.path if 'dist-packages' not in p] + [p for p in sys.path if 'dist-packages' in p]


# =============================================================================
# CRITICAL FIX APPLIED: Restored correct transformers import and config logic.
# Date: 2024-10-27
# Action: Force commit trigger to resolve ModuleNotFoundError in Colab.
# =============================================================================
import os
import sys
import argparse
import subprocess
import json
import glob
import hashlib
import torch
# FIXED IMPORT: Ensuring Trainer is correctly imported from transformers
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

# FIXED IMPORT: Restoring config imports that were missing
from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator_turbo, data_collator_standart
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.preprocess_file_based import preprocess_dataset_file_based
from src.preprocess_json import preprocess_dataset_json_based
from src.utils import setup_logger, check_pretrained_models
from src.inference_callback import InferenceCallback

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxFinetune")


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_preprocessed_data(cfg) -> bool:
    """
    Verify if existing preprocessed data is valid.
    
    Checks:
    1. preprocess_report.json exists
    2. total_files count matches current source files
    3. first_file hash matches current first file
    
    Returns True if verification passes (can skip preprocessing), False otherwise.
    """
    report_path = os.path.join(cfg.preprocessed_dir, "preprocess_report.json")
    
    if not os.path.exists(report_path):
        logger.info("No preprocess_report.json found, will run preprocessing")
        return False
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load preprocess_report.json: {e}, will run preprocessing")
        return False
    
    # Count current source files based on dataset type
    if cfg.json_format:
        if not os.path.exists(cfg.metadata_path):
            logger.warning("Metadata file not found, will run preprocessing")
            return False
        # For JSON format, we trust the report's total_files since we can't easily recount without re-reading
        current_file_count = report.get("total_files", 0)
    elif cfg.ljspeech:
        if not os.path.exists(cfg.csv_path):
            logger.warning("CSV file not found, will run preprocessing")
            return False
        import pandas as pd
        try:
            data = pd.read_csv(cfg.csv_path, sep="|", header=None, quoting=3)
            current_file_count = len(data)
        except Exception as e:
            logger.warning(f"Failed to read CSV: {e}, will run preprocessing")
            return False
    else:
        # File-based format
        wav_files = sorted(glob.glob(os.path.join(cfg.wav_dir, "*.wav")))
        current_file_count = len(wav_files)
    
    stored_file_count = report.get("total_files", 0)
    
    # Check 1: File count
    if current_file_count != stored_file_count:
        logger.info(f"File count mismatch: current={current_file_count}, stored={stored_file_count}, will run preprocessing")
        return False
    
    if current_file_count == 0:
        logger.warning("No files to process")
        return False
    
    # Check 2: First file hash
    stored_first_filename = report.get("first_file", {}).get("filename", "")
    stored_first_hash = report.get("first_file", {}).get("hash", "")
    
    if not stored_first_filename or not stored_first_hash:
        logger.warning("First file info missing in report, will run preprocessing")
        return False
    
    # Construct path to first wav file
    first_wav_path = os.path.join(cfg.wav_dir, stored_first_filename)
    
    if not os.path.exists(first_wav_path):
        logger.warning(f"First file not found: {first_wav_path}, will run preprocessing")
        return False
    
    current_first_hash = compute_file_hash(first_wav_path)
    
    if current_first_hash != stored_first_hash:
        logger.info(f"First file hash mismatch, will run preprocessing")
        return False
    
    logger.info(f"Verification passed: {stored_file_count} files, first file '{stored_first_filename}' hash matches")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Chatterbox Finetuning Script")
    parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        help="Path of checkpoint to resume training from"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="Project name for organizing dataset and outputs"
    )
    return parser.parse_args()


def load_config_from_json(config_path="config.json"):
    """Load config values from JSON file if it exists."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def main():
    args = parse_args()
    
    # Create/update colab_config_override.json if project_name is provided via CLI
    if args.project_name is not None and args.project_name != "":
        override_file = "colab_config_override.json"
        overrides = {}
        if os.path.exists(override_file):
            try:
                with open(override_file, 'r') as f:
                    overrides = json.load(f)
            except Exception:
                pass
        overrides["project_name"] = args.project_name
        with open(override_file, 'w') as f:
            json.dump(overrides, f, indent=2)
        logger.info(f"Updated {override_file} with project_name: {args.project_name}")
    
    # Initialize config - TrainConfig will read from colab_config_override.json automatically
    cfg = TrainConfig()
    
    # Validate that project_name is set
    if not cfg.project_name:
        logger.error("project_name is not set! Please provide it via --project_name argument or ensure colab_config_override.json contains a valid project_name.")
        sys.exit(1)

    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SELECT THE CORRECT ENGINE CLASS
    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS

    is_resuming = args.resume is not None

    if is_resuming:
        adapter_path = os.path.join(cfg.output_dir, "new_lang_adapter")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter directory not found at {adapter_path}. Cannot resume LoRA training without adapter config.")
        
        tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")
        pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
        original_t3_config = tts_engine_original.t3.hp

        new_t3_config = original_t3_config
        new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
        if hasattr(new_t3_config, "use_cache"):
            new_t3_config.use_cache = False
        else:
            setattr(new_t3_config, "use_cache", False)

        new_t3_model = T3(hp=new_t3_config)
        new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)

        if cfg.is_turbo:
            # For turbo models using GPT2, we need to remove word embeddings
            # In newer transformers versions (4.50+), wte/wpe are properties and cannot be deleted
            # Skip deletion as it's not needed for fine-tuning with custom embeddings
            pass

        del tts_engine_original
        del pretrained_t3_state_dict

        tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
        tts_engine_new.t3 = new_t3_model

        for param in tts_engine_new.ve.parameters():
            param.requires_grad = False
        for param in tts_engine_new.s3gen.parameters():
            param.requires_grad = False

        if cfg.is_lora:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.turbo_lora_target_modules if cfg.is_turbo else cfg.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                modules_to_save=cfg.lora_modules_to_save,
            )
            tts_engine_new.t3 = get_peft_model(tts_engine_new.t3, peft_config)
            tts_engine_new.t3.load_adapter(adapter_path)
        else:
            tts_engine_new.t3.train()
            for param in tts_engine_new.t3.parameters():
                param.requires_grad = True
        
    else:
        tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

        pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
        original_t3_config = tts_engine_original.t3.hp

        new_t3_config = original_t3_config
        new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

        if hasattr(new_t3_config, "use_cache"):
            new_t3_config.use_cache = False
        else:
            setattr(new_t3_config, "use_cache", False)

        new_t3_model = T3(hp=new_t3_config)
        new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)

        if cfg.is_turbo:
            # For turbo models using GPT2, we need to remove word embeddings
            # In newer transformers versions (4.50+), wte/wpe are properties and cannot be deleted
            # Skip deletion as it's not needed for fine-tuning with custom embeddings
            pass

        del tts_engine_original
        del pretrained_t3_state_dict

        tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
        tts_engine_new.t3 = new_t3_model

        for param in tts_engine_new.ve.parameters():
            param.requires_grad = False

        for param in tts_engine_new.s3gen.parameters():
            param.requires_grad = False

        if cfg.is_lora:
            for param in tts_engine_new.t3.parameters():
                param.requires_grad = False
                
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.turbo_lora_target_modules if cfg.is_turbo else cfg.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                modules_to_save=cfg.lora_modules_to_save,
            )

            tts_engine_new.t3 = get_peft_model(tts_engine_new.t3, peft_config)
            tts_engine_new.t3.print_trainable_parameters()

        else:
            tts_engine_new.t3.train()
            for param in tts_engine_new.t3.parameters():
                param.requires_grad = True

    # 7. PREPROCESSING
    # Check if existing preprocessed data is valid
    if verify_preprocessed_data(cfg):
        logger.info("Verification passed, skipping preprocessing and using existing .pt files")
    else:
        logger.info("Running preprocessing...")
        if cfg.ljspeech:
            preprocess_dataset_ljspeech(cfg, tts_engine_new)
        elif cfg.json_format:
            preprocess_dataset_json_based(cfg, tts_engine_new)
        else:
            preprocess_dataset_file_based(cfg, tts_engine_new)

    # 8. DATASET & WRAPPER
    train_ds = ChatterboxDataset(cfg)

    trainer_callbacks = []
    if cfg.is_inference:
        inference_cb = InferenceCallback(cfg)
        trainer_callbacks.append(inference_cb)

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    if cfg.is_turbo:
        selected_collator = data_collator_turbo
    else:
        selected_collator = data_collator_standart

    # 9. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=False,
        bf16=True,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=True,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        resume_from_checkpoint=args.resume,
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=selected_collator,
        callbacks=trainer_callbacks,
    )

    trainer.train()

    # 10. SAVE FINAL MODEL
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.is_lora:
        save_path = os.path.join(cfg.output_dir, "new_lang_adapter")
        tts_engine_new.t3.save_pretrained(save_path)
        
        if cfg.is_merge_lora:
            subprocess.run([sys.executable, "merge_lora.py"], check=True)
    else:
        filename = "t3_turbo_finetuned.safetensors" if cfg.is_turbo else "t3_finetuned.safetensors"
        final_model_path = os.path.join(cfg.output_dir, filename)
        save_file(tts_engine_new.t3.state_dict(), final_model_path)
        logger.info(f"Full model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
