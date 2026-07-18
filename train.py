import os
import sys
import argparse
import subprocess
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

from src.config import TrainConfig, should_run_preprocessing
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
        default="",
        help="Project name for organizing dataset and outputs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Override config with command-line arguments if provided
    config_kwargs = {}
    if args.project_name:
        config_kwargs["project_name"] = args.project_name
    
    cfg = TrainConfig(**config_kwargs)

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
            if hasattr(new_t3_model.tfmr, "wte"):
                del new_t3_model.tfmr.wte

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
            if hasattr(new_t3_model.tfmr, "wte"):
                del new_t3_model.tfmr.wte

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
    if should_run_preprocessing(cfg):
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
