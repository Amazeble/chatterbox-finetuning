import os
import torch
import numpy as np
import soundfile as sf
import random
import re
import argparse
from safetensors.torch import load_file

from src.model import resize_and_load_t3_weights
from src.utils import setup_logger, trim_silence_with_vad
from src.config import TrainConfig
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3


logger = setup_logger("Chatterbox-Inference")

cfg = TrainConfig()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_TURBO = cfg.is_turbo
IS_LORA = cfg.is_lora
BASE_MODEL_DIR = cfg.model_dir
OUTPUT_DIR = cfg.output_dir

# --- LoRA path ---
LORA_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "new_lang_adapter")

# --- Checkpoints and Generation Params ---
if IS_TURBO:
    if IS_LORA:
        FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, "t3_turbo_finetuned_merged.safetensors")
        PARAMS = {
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }   
    else:
        FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, "t3_turbo_finetuned.safetensors")
        PARAMS = {
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }
else:
    if IS_LORA:
        FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, "t3_finetuned_merged.safetensors")
        PARAMS = {
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }
    else:
        FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, "t3_finetuned.safetensors")
        PARAMS = {
            "temperature": 0.8,
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "repetition_penalty": 1.2,
        }

TEXT_TO_SAY = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı ve şimdi sahip olduğuma göre, sessiz kalmayacağım."
AUDIO_PROMPT = "./speaker_reference/2.wav"
OUTPUT_FILE = "./output.wav"
KEEP_ALL_SILENT = False


# -----------------------------------------------------------------------------
# Engine loaders
# -----------------------------------------------------------------------------

def load_finetuned_engine_lora(device):
    """Loads the engine and restores a LoRA-finetuned T3 (with resized vocab)."""
    from peft import PeftModel

    logger.info(f"Loading in {'TURBO' if IS_TURBO else 'NORMAL'} mode  [LoRA]")
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    EngineClass = ChatterboxTurboTTS if IS_TURBO else ChatterboxTTS

    # 1. Load base model temporarily to grab pretrained weights + config
    temp_engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")
    pretrained_state = temp_engine.t3.state_dict()
    original_config = temp_engine.t3.hp

    # 2. Build new T3 with expanded vocab
    logger.info(f"Initializing new T3 with vocab size: {cfg.new_vocab_size}")
    new_t3_config = original_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False

    new_t3 = T3(hp=new_t3_config)

    # 3. Transfer & resize pretrained weights into new T3
    logger.info("Resizing base weights to fit new vocab...")
    new_t3 = resize_and_load_t3_weights(new_t3, pretrained_state)

    del temp_engine
    del pretrained_state

    # 4. Turbo: remove wte + monkey-patch (required by PEFT)
    if IS_TURBO:
        logger.info("Turbo Mode: Deleting 'wte' and applying monkey patch.")
        if hasattr(new_t3.tfmr, "wte"):
            del new_t3.tfmr.wte


    # 5. Prepare fresh engine and inject rebuilt T3
    tts_engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")
    tts_engine.t3 = new_t3

    # 6. Load LoRA adapters + trained embeddings (modules_to_save)
    if not os.path.exists(LORA_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Adapter path not found: {LORA_CHECKPOINT_PATH}")

    logger.info(f"Loading LoRA adapters & embeddings from: {LORA_CHECKPOINT_PATH}")
    tts_engine.t3 = PeftModel.from_pretrained(
        tts_engine.t3,
        LORA_CHECKPOINT_PATH,
        is_trainable=False,
    )

    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()
    tts_engine.device = device

    return tts_engine


def load_finetuned_engine_full(device):
    """Loads the engine and restores a fully fine-tuned T3 from safetensors."""

    logger.info(f"Loading in {'TURBO' if IS_TURBO else 'NORMAL'} mode  [Full Fine-Tune]")
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    EngineClass = ChatterboxTurboTTS if IS_TURBO else ChatterboxTTS

    tts_engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")

    logger.info(f"Initializing new T3 with vocab size: {cfg.new_vocab_size}")
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size

    new_t3 = T3(hp=t3_config)

    if IS_TURBO:
        logger.info("Turbo Mode: Removing 'wte' layer from new T3 model.")
        if hasattr(new_t3.tfmr, "wte"):
            del new_t3.tfmr.wte

    if not os.path.exists(FINETUNED_WEIGHTS):
        logger.error(f"FATAL: Fine-tuned file not found at {FINETUNED_WEIGHTS}.")
        raise FileNotFoundError(FINETUNED_WEIGHTS)

    logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
    state_dict = load_file(FINETUNED_WEIGHTS, device="cpu")
    new_t3.load_state_dict(state_dict, strict=True)
    logger.info("Fine-tuned weights loaded successfully.")

    tts_engine.t3 = new_t3
    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()
    tts_engine.device = device

    return tts_engine


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def generate_sentence_audio(engine, text, prompt_path, keep_all_silent=False, **kwargs):
    """Generates audio for a single sentence and trims silence if not keeping all silent."""
    try:
        wav_tensor = engine.generate(text=text, audio_prompt_path=prompt_path, **kwargs)
        if isinstance(wav_tensor, tuple):
            wav_tensor = wav_tensor[0]
        wav_np = wav_tensor.squeeze().cpu().numpy()
        
        if not keep_all_silent:
            trimmed_wav = trim_silence_with_vad(wav_np, engine.sr)
            return engine.sr, trimmed_wav
        else:
            return engine.sr, wav_np
    except Exception as e:
        logger.error(f"Error generating sentence '{text[:30]}...': {e}")
        return 24000, np.zeros(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Chatterbox Inference Script")
    parser.add_argument(
        "--keep-all-silent",
        action="store_true",
        help="Keep all silence in the generated audio. When set, no silence will be removed by VAD trimming."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set global flag based on command line argument
    global KEEP_ALL_SILENT
    KEEP_ALL_SILENT = args.keep_all_silent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference running on: {device}")
    logger.info(f"Training strategy: {'LoRA' if IS_LORA else 'Full Fine-Tune'}")
    logger.info(f"Keep all silent: {KEEP_ALL_SILENT}")

    if IS_LORA:
        engine = load_finetuned_engine_lora(device)
    else:
        engine = load_finetuned_engine_full(device)

    sentences = re.split(r'(?<=[.?!])\s+', TEXT_TO_SAY.strip())
    sentences = [s for s in sentences if s.strip()]

    logger.info(f"Found {len(sentences)} sentences to synthesize.")

    all_chunks = []
    sample_rate = 24000

    set_seed(42)

    for i, sent in enumerate(sentences):
        logger.info(f"Synthesizing ({i+1}/{len(sentences)}): {sent}")
        sr, audio_chunk = generate_sentence_audio(engine, sent, AUDIO_PROMPT, keep_all_silent=KEEP_ALL_SILENT, **PARAMS)

        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            pause_samples = int(sr * 0.2)
            all_chunks.append(np.zeros(pause_samples, dtype=np.float32))

    if all_chunks:
        final_audio = np.concatenate(all_chunks)
        sf.write(OUTPUT_FILE, final_audio, sample_rate)
        logger.info(f"Result saved to: {OUTPUT_FILE}")
    else:
        logger.error("No audio was generated.")


if __name__ == "__main__":
    main()