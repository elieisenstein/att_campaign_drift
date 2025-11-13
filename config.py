"""
Configuration file for SMS Campaign Drift Detection System
All parameters and constants centralized for easy experimentation and deployment
"""

import os
from typing import Dict, Any

# ====================
# FILE PATHS & STORAGE
# ====================

# Artifacts directory for all outputs
ART_DIR = "./artifacts"

# Prefix for file naming convention
PREFIX = "week_synth"

# Core data paths - derived from prefix
META_CSV = os.path.join(ART_DIR, f"{PREFIX}.csv")
VEC_NPY = os.path.join(ART_DIR, f"{PREFIX}.npy")

# Campaign artifacts
CENTROIDS_PATH = os.path.join(ART_DIR, f"{PREFIX}_campaign_centroids.npy")
CAMPAIGNS_CSV = os.path.join(ART_DIR, f"{PREFIX}_campaigns.csv")
CAMPAIGN_EXAMPLES_CSV = os.path.join(ART_DIR, f"{PREFIX}_campaign_examples.csv")
POINTS_CSV = os.path.join(ART_DIR, f"{PREFIX}_points.csv")

# UMAP model persistence
UMAP_MODEL_PATH = os.path.join(ART_DIR, "umap_model.pkl")

# Prototype outputs for new data
PROTO_META_CSV = os.path.join(ART_DIR, f"{PREFIX}_new_prototypes.csv")
PROTO_NPY = os.path.join(ART_DIR, f"{PREFIX}_new_prototypes.npy")
PROTO_ASSIGNMENTS_CSV = os.path.join(ART_DIR, f"{PREFIX}_new_prototypes_assignments.csv")
PROTO_ASSIGNMENTS_UPSERTED_CSV = os.path.join(ART_DIR, f"{PREFIX}_new_prototypes_assignments_upserted.csv")

# Visualization outputs
UMAP_PLOT_PATH = os.path.join(ART_DIR, "umap_hdbscan_campaigns.png")
DEBUG_UMAP_PATH = os.path.join(ART_DIR, "stage6_debug_umap.png")

# ====================
# DATA SOURCES
# ====================

# Input data files
SYNTHETIC_DATA_PATH = "./artifacts/synthetic_one_originator.csv"
NEW_BATCH_PATH = "./artifacts/mixed_160_orig__40_new.csv"

# ====================
# MODEL CONFIGURATION
# ====================

# Embedding model (local path or HuggingFace model name)
LOCAL_MODEL = r"C:/models/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Embedding parameters
EMBED_BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True

# ====================
# CLUSTERING PARAMETERS
# ====================

# UMAP dimensionality reduction (for visualization)
UMAP_PARAMS: Dict[str, Any] = dict(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    force_approximation_algorithm=False,
    transform_seed=42,
)

# HDBSCAN clustering (reference profile building)
HDBSCAN_PARAMS: Dict[str, Any] = dict(
    min_cluster_size=5,
    min_samples=2,
    metric="euclidean",
)

# HDBSCAN for unknown prototype clustering (Stage 3)
HDBSCAN_UNKNOWN_PARAMS: Dict[str, Any] = dict(
    min_cluster_size=3,
    min_samples=1,
    metric="euclidean"  # embeddings are unit-normalized -> euclidean ~ cosine
)

# ====================
# DRIFT DETECTION THRESHOLDS
# ====================

# Similarity threshold for campaign assignment (per SOW)
SIM_THRESHOLD = 0.6

# Centroid similarity threshold for conservative assignment
CENTROID_SIM_THRESHOLD = 0.6

# Minimum coverage before triggering unknown clustering
MIN_COVERAGE = 0.8

# Minimum number of unknown prototypes to consider clustering
MIN_UNASSIGNED = 5

# Minimum cluster size to keep (for unknown clustering)
MIN_CLUSTER_MEMBERS = 3

# ====================
# CAMPAIGN PROFILING
# ====================

# Number of nearest samples to use as exemplars
N_NEAREST = 5

# Number of samples to pass to LLM for summarization
N_SAMPLES = 10

# ====================
# LLM CONFIGURATION
# ====================

# LLM model for campaign name generation
LLM_MODEL = "gpt-4o-mini"

# LLM generation parameters
LLM_TEMPERATURE = 0.0
LLM_MAX_WORDS = 5

# ====================
# TEXT PROCESSING
# ====================

# Random seed for hash functions
HASH_SEED = 0

# Text normalization rules (can be expanded based on sms_norm module)
NORMALIZATION_CONFIG = {
    'remove_urls': True,
    'lowercase': True,
    'remove_special_chars': True,
    'normalize_whitespace': True,
}

# ====================
# COLUMN NAMES
# ====================

# Standard column names used throughout the pipeline
COLUMNS = {
    # Input columns
    'raw_text': 'raw_text',
    'text': 'text',
    'originator_id': 'originator_id',
    'message_id': 'message_id',
    
    # Normalized/processed columns
    'normalized_text': 'normalized_text',
    'template_hash_xx64': 'template_hash_xx64',
    
    # Assignment columns
    'label': 'label',
    'cluster_label': 'cluster_label',
    'campaign_name': 'campaign_name',
    'status': 'status',
    
    # Scoring columns
    'assigned_campaign_score': 'assigned_campaign_score',
    'sim_score': 'sim_score',
    
    # Metadata columns
    'count_in_window': 'count_in_window',
    'window_start': 'window_start',
    'window_end': 'window_end',
    'timestamp': 'timestamp',
    
    # Proposed columns (for unknown clustering)
    'proposed_cluster_id': 'proposed_cluster_id',
    'proposed_campaign_name': 'proposed_campaign_name',
    'proposed_campaign_score': 'proposed_campaign_score',
    
    # UMAP coordinates
    'umap_x': 'umap_x',
    'umap_y': 'umap_y',
}

# ====================
# ENVIRONMENT CONFIGURATION
# ====================

# Environment-based configuration (can be extended)
import os as _os
ENV = _os.getenv('ENV', 'dev')

if ENV == 'production':
    # Production overrides
    ART_DIR = "/mnt/data/artifacts"
    MIN_CLUSTER_SIZE = 10
    SIM_THRESHOLD = 0.7
elif ENV == 'staging':
    # Staging overrides
    ART_DIR = "/tmp/artifacts"
    LLM_TEMPERATURE = 0.1
else:
    # Development settings (already defined above)
    pass

# ====================
# VALIDATION
# ====================

def validate_config():
    """Validate configuration parameters"""
    assert 0 <= SIM_THRESHOLD <= 1, "SIM_THRESHOLD must be between 0 and 1"
    assert 0 <= CENTROID_SIM_THRESHOLD <= 1, "CENTROID_SIM_THRESHOLD must be between 0 and 1"
    assert 0 <= MIN_COVERAGE <= 1, "MIN_COVERAGE must be between 0 and 1"
    assert MIN_UNASSIGNED > 0, "MIN_UNASSIGNED must be positive"
    assert MIN_CLUSTER_MEMBERS > 0, "MIN_CLUSTER_MEMBERS must be positive"
    assert N_NEAREST > 0, "N_NEAREST must be positive"
    assert N_SAMPLES > 0, "N_SAMPLES must be positive"
    assert LLM_MAX_WORDS > 0, "LLM_MAX_WORDS must be positive"
    assert EMBED_BATCH_SIZE > 0, "EMBED_BATCH_SIZE must be positive"
    assert UMAP_PARAMS['n_neighbors'] > 0, "UMAP n_neighbors must be positive"
    assert HDBSCAN_PARAMS['min_cluster_size'] > 0, "HDBSCAN min_cluster_size must be positive"
    
    # Ensure directories exist or can be created
    if not _os.path.exists(ART_DIR):
        print(f"Warning: ART_DIR '{ART_DIR}' does not exist. It will be created when needed.")
    
    return True

# Run validation when module is imported (optional)
# validate_config()