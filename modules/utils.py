# modules/utils.py
# Full corrected file with new persistence function added at the end.

import logging
import os
import shutil
import time
from datetime import datetime
import re
from pathlib import Path

CONFIG_FILENAME = "config.py"

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *( [logging.FileHandler(log_file)] if log_file else [] )
        ]
    )
    
    return logging.getLogger(__name__)

def manage_storage(folder_path, max_size_mb):
    """Manage storage space by cleaning old files"""
    try:
        if not os.path.exists(folder_path):
            return
            
        # Calculate current folder size
        total_size = 0
        files_info = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_mtime = os.path.getmtime(file_path)
                files_info.append((file_path, file_size, file_mtime))
        
        size_mb = total_size / (1024 * 1024)
        
        # Cleanup if over limit
        if size_mb > max_size_mb:
            # Sort by modification time (oldest first)
            files_info.sort(key=lambda x: x[2])
            
            deleted_size = 0
            target_size = max_size_mb * 0.8 * 1024 * 1024  # Target 80% of max
            
            for file_path, file_size, _ in files_info:
                if total_size - deleted_size <= target_size:
                    break
                    
                try:
                    os.remove(file_path)
                    deleted_size += file_size
                    logging.info(f"Cleaned up: {os.path.basename(file_path)}")
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}: {e}")
                    
            logging.info(f"Storage cleanup: Deleted {deleted_size / (1024 * 1024):.2f} MB")
            
    except Exception as e:
        logging.error(f"Storage management failed: {e}")

def create_timestamp():
    """Create standardized timestamp"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_directory(directory_path):
    """Ensure directory exists"""
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return 0

def safe_delete(file_path):
    """Safely delete a file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        logging.error(f"Safe delete failed for {file_path}: {e}")
    return False

def format_duration(seconds):
    """Format duration in seconds to readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# NEW FUNCTION: For persisting keg count to config.py
def save_default_keg_count(value: int):
    """Safely update DEFAULT_KEG_COUNT in config.py"""
    config_path = Path(__file__).parent.parent / CONFIG_FILENAME  # Assumes modules/ is subdir of root
    if not config_path.exists():
        logging.warning("config.py not found!")
        return False
    
    content = config_path.read_text(encoding='utf-8')
    # Regex matches: DEFAULT_KEG_COUNT = 6 (with optional whitespace)
    pattern = r'(DEFAULT_KEG_COUNT\s*=\s*)(\d+)'
    new_content = re.sub(pattern, rf'\g<1>{value}', content)
    
    config_path.write_text(new_content, encoding='utf-8')
    logging.info(f"Config updated: DEFAULT_KEG_COUNT = {value}")
    return True

# Add these functions to utils.py
def save_last_batch(batch_number: str):
    """Save the last batch number to file"""
    try:
        config_path = Path(__file__).parent.parent / CONFIG_FILENAME
        config_dir = config_path.parent
        
        # Create batch memory file
        batch_file = config_dir / "last_batch.txt"
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_number)
        
        logging.info(f"Saved last batch: {batch_number}")
        return True
    except Exception as e:
        logging.error(f"Failed to save last batch: {e}")
        return False

def load_last_batch() -> str:
    """Load the last batch number from file"""
    try:
        config_path = Path(__file__).parent.parent / CONFIG_FILENAME
        config_dir = config_path.parent
        batch_file = config_dir / "last_batch.txt"
        
        if batch_file.exists():
            with open(batch_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return content
        
        # Return default if file doesn't exist or is empty
        return "BATCH-001"
    except Exception as e:
        logging.error(f"Failed to load last batch: {e}")
        return "BATCH-001"