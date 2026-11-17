    learning_rate: float = 1e-4 if is_lora else 1e-5
    num_epochs: int = 10 if is_lora else 20
    save_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 5
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
