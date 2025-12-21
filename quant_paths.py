# quant_paths.py - ENHANCED VERSION
from pathlib import Path
from datetime import datetime
import pandas as pd

def get_project_root():
    """Get project root directory"""
    return Path(__file__).resolve().parent

def get_results_file_path():
    """Get path to results Excel file"""
    return get_project_root() / "number prediction learn.xlsx"

def get_predictions_dir(subfolder=None):
    """Get predictions directory"""
    base_dir = get_project_root() / "predictions"
    if subfolder:
        return base_dir / subfolder
    return base_dir

def get_bet_engine_dir():
    """Get bet engine directory"""
    return get_predictions_dir("bet_engine")

def get_logs_dir():
    """Get logs directory"""
    return get_project_root() / "logs"

def get_performance_logs_dir():
    """Get performance logs directory"""
    return get_logs_dir() / "performance"

def get_bet_plan_master_path(date_str):
    """Get bet plan master file path for date"""
    date_str_clean = date_str.replace("-", "")
    return get_bet_engine_dir() / f"bet_plan_master_{date_str_clean}.xlsx"

def get_final_bet_plan_path(date_str):
    """Get final bet plan file path for date"""
    date_str_clean = date_str.replace("-", "")
    return get_bet_engine_dir() / f"final_bet_plan_{date_str_clean}.xlsx"

def get_live_bet_sheet_path(date_str):
    """Get live bet sheet file path for date"""
    date_str_clean = date_str.replace("-", "")
    return get_bet_engine_dir() / f"live_bet_sheet_{date_str_clean}.xlsx"

def find_latest_bet_plan_master():
    """Find latest bet plan master file"""
    bet_engine_dir = get_bet_engine_dir()
    bet_files = list(bet_engine_dir.glob("bet_plan_master_*.xlsx"))
    if bet_files:
        return max(bet_files, key=lambda x: x.stat().st_mtime)
    return None

def find_latest_final_bet_plan():
    """Find latest final bet plan file"""
    bet_engine_dir = get_bet_engine_dir()
    bet_files = list(bet_engine_dir.glob("final_bet_plan_*.xlsx"))
    if bet_files:
        return max(bet_files, key=lambda x: x.stat().st_mtime)
    return None

def parse_date_from_filename(filename):
    """Parse date from filename"""
    try:
        # Handle different filename patterns
        if "bet_plan_master_" in filename:
            date_str = filename.replace("bet_plan_master_", "").replace(".xlsx", "")
        elif "final_bet_plan_" in filename:
            date_str = filename.replace("final_bet_plan_", "").replace(".xlsx", "")
        elif "live_bet_sheet_" in filename:
            date_str = filename.replace("live_bet_sheet_", "").replace(".xlsx", "")
        else:
            return None
        
        if len(date_str) == 8:  # YYYYMMDD format
            return datetime.strptime(date_str, "%Y%m%d").date()
        else:
            return None
    except:
        return None

def get_base_dir():
    """Alias for get_project_root for compatibility"""
    return get_project_root()

# Compatibility functions for existing code
def get_bet_plans_dir():
    """Compatibility function"""
    return get_bet_engine_dir()

def get_performance_logs_dir():
    """Compatibility function"""
    return get_project_root() / "logs" / "performance"