import os
from safetensors.torch import save_file
from peft import PeftModel

from src.config import TrainConfig
from src.model import resize_and_load_t3_weights
from src.utils import setup_logger

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

logger = setup_logger("ChatterboxMerge")

def main():
    cfg = TrainConfig()
    
    logger.info("--- Starting LoRA Merge Process ---")
    
    base_model_dir = cfg.model_dir
    adapter_dir = os.path.join(cfg.output_dir, "new_lang_adapter")
    

    output_filename = "t3_turbo_finetuned_merged.safetensors" if cfg.is_turbo else "t3_finetuned_merged.safetensors"
    output_file = os.path.join(cfg.output_dir, output_filename)
    
    if not os.path.exists(adapter_dir):
        logger.error(f"Adapter directory not found at {adapter_dir}! Please train the model first.")
        return

    logger.info("1. Loading base model configuration...")
    EngineClass = ChatterboxTurboTTS if cfg.is_turbo else ChatterboxTTS
    tts_engine_base = EngineClass.from_local(base_model_dir, device="cpu")
    
    pretrained_state_dict = tts_engine_base.t3.state_dict()
    t3_config = tts_engine_base.t3.hp
    
    logger.info(f"2. Preparing T3 model with new vocab size: {cfg.new_vocab_size}...")
    t3_config.text_tokens_dict_size = cfg.new_vocab_size
    new_t3_model = T3(hp=t3_config)
    
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_state_dict)
    
    if cfg.is_turbo and hasattr(new_t3_model.tfmr, "wte"):
        logger.info("Deleting tfmr.wte to match training architecture...")
        del new_t3_model.tfmr.wte

    del tts_engine_base
    del pretrained_state_dict

    logger.info(f"3. Loading LoRA adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(new_t3_model, adapter_dir)


    logger.info("4. Merging LoRA weights into base model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    logger.info(f"5. Saving standalone merged model to {output_file}...")
    save_file(merged_model.state_dict(), output_file)
    
    logger.info("--- MERGE COMPLETE ---")

if __name__ == "__main__":
    main()