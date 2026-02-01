"""Structure definitions for pelvis OAR segmentation.

This module defines structure names, aliases, expected volume ranges, 
display colors, and difficulty ratings for pelvis organ at risk (OAR) segmentation.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum


class StructureDifficulty(Enum):
    """Difficulty rating for structure segmentation."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


# Structure name mapping - canonical name to list of possible aliases
STRUCTURE_ALIASES: Dict[str, List[str]] = {
    "bladder": [
        "bladder", "Bladder", "BLADDER",
        "bladder_o", "Bladder_O", "BLADDER_O",
        "vesica", "Vesica", "VESICA",
        "urinary_bladder", "Urinary_Bladder", "URINARY_BLADDER",
        "blad", "Blad", "BLAD"
    ],
    "rectum": [
        "rectum", "Rectum", "RECTUM",
        "rectum_o", "Rectum_O", "RECTUM_O",
        "rect", "Rect", "RECT"
    ],
    "bowel": [
        "bowel", "Bowel", "BOWEL",
        "bowel_bag", "Bowel_Bag", "BOWEL_BAG",
        "bowelbag", "BowelBag", "BOWELBAG",
        "small_bowel", "Small_Bowel", "SMALL_BOWEL",
        "bowel_o", "Bowel_O", "BOWEL_O"
    ],
    "femoral_head_left": [
        "femoral_head_left", "Femoral_Head_Left", "FEMORAL_HEAD_LEFT",
        "femoral_head_l", "Femoral_Head_L", "FEMORAL_HEAD_L",
        "femur_head_left", "Femur_Head_Left", "FEMUR_HEAD_LEFT",
        "femur_head_l", "Femur_Head_L", "FEMUR_HEAD_L",
        "femoralhead_l", "FemoralHead_L", "FEMORALHEAD_L",
        "femur_l", "Femur_L", "FEMUR_L"
    ],
    "femoral_head_right": [
        "femoral_head_right", "Femoral_Head_Right", "FEMORAL_HEAD_RIGHT",
        "femoral_head_r", "Femoral_Head_R", "FEMORAL_HEAD_R",
        "femur_head_right", "Femur_Head_Right", "FEMUR_HEAD_RIGHT",
        "femur_head_r", "Femur_Head_R", "FEMUR_HEAD_R",
        "femoralhead_r", "FemoralHead_R", "FEMORALHEAD_R",
        "femur_r", "Femur_R", "FEMUR_R"
    ],
    "penile_bulb": [
        "penile_bulb", "Penile_Bulb", "PENILE_BULB",
        "penilebulb", "PenileBulb", "PENILEBULB",
        "bulb", "Bulb", "BULB",
        "penis_bulb", "Penis_Bulb", "PENIS_BULB"
    ],
    "vaginal_canal": [
        "vaginal_canal", "Vaginal_Canal", "VAGINAL_CANAL",
        "vagina", "Vagina", "VAGINA",
        "vaginal", "Vaginal", "VAGINAL",
        "vaginalcanal", "VaginalCanal", "VAGINALCANAL"
    ],
    "lymph_nodes": [
        "lymph_nodes", "Lymph_Nodes", "LYMPH_NODES",
        "lymphnodes", "LymphNodes", "LYMPHNODES",
        "nodes", "Nodes", "NODES",
        "ln", "LN"
    ]
}


# Expected volume ranges for each structure (in cm³)
# Format: (min_volume, max_volume, typical_min, typical_max)
# typical range is used for quality assessment warnings
STRUCTURE_VOLUME_RANGES: Dict[str, Tuple[float, float, float, float]] = {
    "bladder": (10.0, 1000.0, 100.0, 500.0),
    "rectum": (20.0, 300.0, 50.0, 150.0),
    "bowel": (200.0, 3000.0, 500.0, 1500.0),
    "femoral_head_left": (30.0, 150.0, 50.0, 100.0),
    "femoral_head_right": (30.0, 150.0, 50.0, 100.0),
    "penile_bulb": (1.0, 20.0, 3.0, 10.0),
    "vaginal_canal": (5.0, 50.0, 10.0, 30.0),
    "lymph_nodes": (50.0, 1000.0, 100.0, 500.0)
}


# Structure colors for DICOM RTSTRUCT display (RGB values 0-255)
# Colors chosen to match Eclipse TPS conventions where possible
STRUCTURE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "bladder": (255, 255, 0),      # Yellow
    "rectum": (139, 69, 19),        # Brown
    "bowel": (255, 165, 0),         # Orange
    "femoral_head_left": (0, 255, 0),    # Green
    "femoral_head_right": (0, 255, 0),   # Green
    "penile_bulb": (255, 0, 255),   # Magenta
    "vaginal_canal": (255, 192, 203), # Pink
    "lymph_nodes": (0, 128, 255)    # Light Blue
}


# Structure difficulty ratings for training prioritization
STRUCTURE_DIFFICULTY: Dict[str, StructureDifficulty] = {
    "bladder": StructureDifficulty.EASY,
    "rectum": StructureDifficulty.EASY,
    "bowel": StructureDifficulty.HARD,
    "femoral_head_left": StructureDifficulty.MEDIUM,
    "femoral_head_right": StructureDifficulty.MEDIUM,
    "penile_bulb": StructureDifficulty.VERY_HARD,
    "vaginal_canal": StructureDifficulty.HARD,
    "lymph_nodes": StructureDifficulty.VERY_HARD
}


# Label mapping for nnU-Net (0 is background, 1-N are structures)
STRUCTURE_LABELS: Dict[str, int] = {
    "bladder": 1,
    "rectum": 2,
    "bowel": 3,
    "femoral_head_left": 4,
    "femoral_head_right": 5,
    "penile_bulb": 6,
    "vaginal_canal": 7,
    "lymph_nodes": 8
}


def get_canonical_name(structure_name: str) -> Optional[str]:
    """
    Get the canonical structure name from a possible alias.
    
    Args:
        structure_name: Structure name as it appears in RTSTRUCT
        
    Returns:
        Canonical structure name if found, None otherwise
        
    Example:
        >>> get_canonical_name("Bladder_O")
        'bladder'
        >>> get_canonical_name("Unknown_Structure")
        None
    """
    structure_name_stripped = structure_name.strip()
    
    for canonical_name, aliases in STRUCTURE_ALIASES.items():
        if structure_name_stripped in aliases:
            return canonical_name
    
    return None


def get_structure_label(structure_name: str) -> Optional[int]:
    """
    Get the nnU-Net label for a structure.
    
    Args:
        structure_name: Structure name (canonical or alias)
        
    Returns:
        Label integer if found, None otherwise
    """
    canonical_name = get_canonical_name(structure_name)
    if canonical_name:
        return STRUCTURE_LABELS.get(canonical_name)
    return None


def get_structure_color(structure_name: str) -> Optional[Tuple[int, int, int]]:
    """
    Get the RGB color for a structure.
    
    Args:
        structure_name: Structure name (canonical or alias)
        
    Returns:
        RGB tuple (0-255 range) if found, None otherwise
    """
    canonical_name = get_canonical_name(structure_name)
    if canonical_name:
        return STRUCTURE_COLORS.get(canonical_name)
    return None


def get_volume_range(structure_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Get the expected volume range for a structure.
    
    Args:
        structure_name: Structure name (canonical or alias)
        
    Returns:
        Tuple of (min_volume, max_volume, typical_min, typical_max) if found, None otherwise
    """
    canonical_name = get_canonical_name(structure_name)
    if canonical_name:
        return STRUCTURE_VOLUME_RANGES.get(canonical_name)
    return None


def get_all_canonical_names() -> List[str]:
    """
    Get list of all canonical structure names.
    
    Returns:
        List of canonical structure names
    """
    return list(STRUCTURE_ALIASES.keys())


def get_all_labels() -> Dict[str, int]:
    """
    Get the complete label mapping for nnU-Net.
    
    Returns:
        Dictionary mapping canonical structure names to label integers
    """
    return STRUCTURE_LABELS.copy()
