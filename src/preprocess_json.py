
import os
import json
import glob
import torch
import torchaudio
from tqdm import tqdm
import hashlib

from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger
from src.config import TrainConfig

logger = setup_logger(__name__)


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def preprocess_dataset_json_based(config, tts_engine: ChatterboxTTS, continue_mode: bool = False):
    
    """
    Reads metadata from JSON file, processes audio-text pairs, and saves them as .pt.
    Structure:
    - JSON contains: id, text, formatted_text, etc.
    - Audio files: {wav_dir}/{id}.wav
    """
    
    os.makedirs(config.preprocessed_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_engine.ve.to(device)
    tts_engine.s3gen.to(device)
    tts_engine.ve.eval()
    tts_engine.s3gen.eval()
    

    if not os.path.exists(config.metadata_path):
        logger.error(f"ERROR: Metadata file not found: '{config.metadata_path}'!")
        return
    
    with open(config.metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if len(metadata) == 0:
        logger.error(f"ERROR: No items found in metadata file!")
        return
    
    # In continue mode, filter out already processed files
    if continue_mode:
        existing_pt_files = set()
        for pt_file in glob.glob(os.path.join(config.preprocessed_dir, "*.pt")):
            existing_pt_files.add(os.path.basename(pt_file).replace(".pt", ""))
        
        original_metadata = metadata
        metadata = [item for item in metadata if item.get("id") not in existing_pt_files]
        skipped_count = len(original_metadata) - len(metadata)
        if skipped_count > 0:
            logger.info(f"Continue mode: Skipping {skipped_count} already processed files, will process {len(metadata)} remaining files")
        else:
            logger.info(f"Continue mode: All {len(original_metadata)} files already processed")
            return
    
    logger.info(f"Processing dataset... Found items in JSON: {len(metadata)}")
    
    success_count = 0
    total_duration_seconds = 0.0
    first_file_info = None
    
    SPEECH_STOP_ID = getattr(tts_engine.t3.hp, 'stop_speech_token', 6562)
    for item in tqdm(metadata, desc="Preprocessing"):
        try:

            file_id = item.get("id")
            raw_text = item.get("text", "")
            
            if not file_id or not raw_text:
                logger.warning(f"Skipping item with missing id or text")
                continue
            

            wav_path = os.path.join(config.wav_dir, f"{file_id}.wav")
            
            if not os.path.exists(wav_path):
                logger.warning(f"Audio file not found, skipping: {file_id}")
                continue
            
            wav, sr = torchaudio.load(wav_path)
            
            # Calculate duration before any resampling
            duration_seconds = wav.shape[1] / sr
            total_duration_seconds += duration_seconds
            
            # Compute hash only for the first successfully processed file
            if first_file_info is None:
                file_hash = compute_file_hash(wav_path)
                first_file_info = {"filename": file_id + ".wav", "hash": file_hash}
            
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            if sr != S3_SR:
                resampler = torchaudio.transforms.Resample(sr, S3_SR)
                wav = resampler(wav)
            
            wav = wav.to(device)
            
            with torch.no_grad():

                wav_np = wav.cpu().squeeze().numpy()
                spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
                speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()
                
                s_tokens, _ = tts_engine.s3gen.tokenizer(wav.unsqueeze(0))
                raw_speech_tokens = s_tokens.squeeze().cpu()
                
                stop_speech_tensor = torch.tensor([SPEECH_STOP_ID], dtype=raw_speech_tokens.dtype)
                speech_tokens = torch.cat([raw_speech_tokens, stop_speech_tensor], dim=0)
                
                
                prompt_samples = int(config.prompt_duration * S3_SR)
                if wav.shape[1] < prompt_samples:
                    prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
                else:
                    prompt_wav = wav[:, :prompt_samples]
                
                p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0))
                prompt_tokens = p_tokens.squeeze().cpu()
            
            clean_text = punc_norm(raw_text)
            
            if config.is_turbo:
                token_output = tts_engine.tokenizer(clean_text, return_tensors="pt")
                raw_text_tokens = token_output.input_ids[0].cpu()
                
                if tts_engine.tokenizer.eos_token_id is not None:
                    text_eos = torch.tensor([tts_engine.tokenizer.eos_token_id], dtype=raw_text_tokens.dtype)
                    text_tokens = torch.cat([raw_text_tokens, text_eos], dim=0)
                else:
                    text_tokens = raw_text_tokens
            
            else:
                text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()
            
            save_path = os.path.join(config.preprocessed_dir, f"{file_id}.pt")
            
            torch.save({
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
                "text_tokens": text_tokens,
            }, save_path)
            
            success_count += 1

            
        except Exception as e:
            logger.error(f"Error ({item.get('id', 'unknown')}): {e}")
            continue
    
    logger.info(f"Preprocessing completed! Success: {success_count}/{len(metadata)}")
    
    # Format total duration as HH:MM:SS
    hours = int(total_duration_seconds // 3600)
    minutes = int((total_duration_seconds % 3600) // 60)
    seconds = int(total_duration_seconds % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    logger.info(f"Total audio duration: {duration_str} ({total_duration_seconds:.2f} seconds)")
    
    # Save preprocessing report with file count, total duration, and first file hash
    if first_file_info or continue_mode:
        report_path = os.path.join(config.preprocessed_dir, "preprocess_report.json")
        
        # In continue mode, load existing report to update totals
        existing_total_files = 0
        existing_total_duration = 0.0
        existing_first_file = None
        if continue_mode and os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    existing_report = json.load(f)
                existing_total_files = existing_report.get("total_files", 0)
                existing_total_duration = existing_report.get("total_duration_seconds", 0.0)
                existing_first_file = existing_report.get("first_file")
                logger.info(f"Loaded existing report: {existing_total_files} files, adding {success_count} new files")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load existing report: {e}, starting fresh")
        
        # Calculate totals
        final_total_files = existing_total_files + success_count
        final_total_duration = existing_total_duration + total_duration_seconds
        
        # Format final duration
        final_hours = int(final_total_duration // 3600)
        final_minutes = int((final_total_duration % 3600) // 60)
        final_seconds = int(final_total_duration % 60)
        final_duration_str = f"{final_hours:02d}:{final_minutes:02d}:{final_seconds:02d}"
        
        # Use existing first_file info if available, otherwise use current
        final_first_file = existing_first_file if existing_first_file else first_file_info
        
        report = {
            "total_files": final_total_files,
            "total_duration_seconds": final_total_duration,
            "total_duration_formatted": final_duration_str,
            "first_file": final_first_file
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Preprocessing report saved to: {report_path}")
    else:
        logger.warning("No files were processed, skipping report generation")
    
    

if __name__ == "__main__":

    cfg = TrainConfig()
    
    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS
    
    logger.info(f"{EngineClass} engine starting...")
    tts_engine = EngineClass.from_local(cfg.model_dir, device="cpu")
    
    preprocess_dataset_json_based(cfg, tts_engine)