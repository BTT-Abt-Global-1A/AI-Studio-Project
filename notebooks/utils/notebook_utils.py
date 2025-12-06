"""
Shared utility functions for notebook environment setup.
Import this at the start of notebooks to handle Colab/local detection.
"""

from pathlib import Path
import os

def setup_environment():
    """
    Detect environment (Colab vs local) and set up project paths.
    
    Returns:
        tuple: (IS_COLAB, PROJECT_ROOT, DATA_DIR, PROCESSED_DIR, MODELS_DIR)
    """
    try:
        import google.colab
        IS_COLAB = True
        print("✅ Running in Google Colab")
        
        # Mount Google Drive
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        
        # Set project root in Drive
        PROJECT_ROOT = Path("/content/drive/MyDrive/AI-Studio-Project")
        
    except ImportError:
        IS_COLAB = False
        print("✅ Running in local environment")
        
        # Assume running from notebooks/milestone_X/ directory
        PROJECT_ROOT = Path("../..").resolve()
    
    # Create directory structure
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Ensure directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Processed data: {PROCESSED_DIR}")
    print(f"📁 Models directory: {MODELS_DIR}")
    
    return IS_COLAB, PROJECT_ROOT, DATA_DIR, PROCESSED_DIR, MODELS_DIR


def check_gpu_availability():
    """
    Check for GPU availability across different frameworks.
    
    Returns:
        dict: GPU information including availability for PyTorch and TensorFlow
    """
    gpu_info = {
        'cuda_available': False,
        'mps_available': False,
        'tensorflow_gpus': 0,
        'pytorch_device': 'cpu'
    }
    
    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['pytorch_device'] = 'cuda'
            print(f"🔥 PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info['mps_available'] = True
            gpu_info['pytorch_device'] = 'mps'
            print("🍎 PyTorch: Apple MPS enabled")
        else:
            print("💻 PyTorch: CPU only")
    except ImportError:
        print("⚠️ PyTorch not installed")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        tf_gpus = tf.config.list_physical_devices("GPU")
        gpu_info['tensorflow_gpus'] = len(tf_gpus)
        if tf_gpus:
            print(f"⚡ TensorFlow: {len(tf_gpus)} GPU(s) detected")
            # Enable memory growth
            for gpu in tf_gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        else:
            print("💻 TensorFlow: CPU only")
    except ImportError:
        print("⚠️ TensorFlow not installed")
    
    return gpu_info


def load_split_data(processed_dir, split_type='month_stratified_splits'):
    """
    Load train/val/test splits from processed directory.
    
    Args:
        processed_dir (Path): Path to processed data directory
        split_type (str): Type of splits ('month_stratified_splits' or other)
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    import pandas as pd
    
    splits_dir = processed_dir / split_type
    
    train_file = splits_dir / "train_data.csv"
    val_file = splits_dir / "val_data.csv"
    test_file = splits_dir / "test_data.csv"
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        raise FileNotFoundError(
            f"❌ Split files not found in {splits_dir}\n"
            "   Run the data splitting notebook first!"
        )
    
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    print(f"✅ Loaded data splits from {splits_dir}")
    print(f"   Training: {train_df.shape}")
    print(f"   Validation: {val_df.shape}")
    print(f"   Test: {test_df.shape}")
    
    return train_df, val_df, test_df


def prepare_features_targets(train_df, val_df, test_df, target_col='outage_occurred', 
                             exclude_cols=None):
    """
    Prepare features and targets from dataframes.
    
    Args:
        train_df, val_df, test_df: DataFrames with features and target
        target_col (str): Name of target column
        exclude_cols (list): Columns to exclude from features
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)
    """
    if exclude_cols is None:
        exclude_cols = [target_col, 'date', 'fips_code', 'month']
    
    # Get common columns across all datasets
    common_cols = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
    
    # Remove excluded columns
    feature_cols = [col for col in common_cols if col not in exclude_cols]
    
    print(f"📊 Using {len(feature_cols)} features")
    print(f"   First 5: {feature_cols[:5]}")
    if len(feature_cols) > 5:
        print(f"   ... and {len(feature_cols) - 5} more")
    
    # Prepare features and targets
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Print class distribution
    print(f"\n📈 Class distribution:")
    print(f"   Training - Outages: {y_train.sum():,} ({y_train.mean():.2%})")
    print(f"   Validation - Outages: {y_val.sum():,} ({y_val.mean():.2%})")
    print(f"   Test - Outages: {y_test.sum():,} ({y_test.mean():.2%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def save_model_artifacts(model, metrics, config, model_dir, model_name):
    """
    Save model, metrics, and configuration.
    
    Args:
        model: Trained model object
        metrics (dict): Dictionary of metric names and values
        config (dict): Model configuration parameters
        model_dir (Path): Directory to save artifacts
        model_name (str): Name prefix for saved files
    """
    import joblib
    import json
    import pandas as pd
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = model_dir / f"{model_name}.pkl"
    joblib.dump(model, model_file)
    print(f"✅ Model saved: {model_file}")
    
    # Save metrics
    metrics_file = model_dir / f"{model_name}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"✅ Metrics saved: {metrics_file}")
    
    # Save config
    config_file = model_dir / f"{model_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved: {config_file}")


def load_model_artifacts(model_dir, model_name):
    """
    Load model, metrics, and configuration.
    
    Args:
        model_dir (Path): Directory containing model artifacts
        model_name (str): Name prefix of saved files
    
    Returns:
        tuple: (model, metrics, config)
    """
    import joblib
    import json
    import pandas as pd
    
    model_dir = Path(model_dir)
    
    # Load model
    model_file = model_dir / f"{model_name}.pkl"
    model = joblib.load(model_file)
    print(f"✅ Model loaded: {model_file}")
    
    # Load metrics
    metrics_file = model_dir / f"{model_name}_metrics.csv"
    metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
    print(f"✅ Metrics loaded: {metrics_file}")
    
    # Load config
    config_file = model_dir / f"{model_name}_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"✅ Config loaded: {config_file}")
    
    return model, metrics, config
