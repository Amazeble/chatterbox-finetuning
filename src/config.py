from dataclasses import dataclass, field
from typing import List, Literal, Optional
import os
import glob


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
