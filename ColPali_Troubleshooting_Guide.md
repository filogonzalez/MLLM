# ColPali Troubleshooting Guide

## Overview

This guide addresses common issues when deploying ColPali in Databricks, particularly focusing on model loading errors and providing robust solutions.

## Common Error: Model File Not Found

### Error Message
```
OSError: vidore/colpaligemma-3b-pt-448-base does not appear to have files named ('model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'). Checkout 'https://huggingface.co/vidore/colpali-v1.3/tree/main'
```

### Root Causes
1. **Model version mismatch**: The model files may have been updated or reorganized
2. **Network connectivity issues**: Intermittent download failures
3. **Cache corruption**: Cached model files may be incomplete
4. **Transformers version incompatibility**: Older versions may not support newer model formats

### Solutions

#### 1. Use Fallback Model Loading (Recommended)
The updated code includes a fallback mechanism that tries multiple model versions:

```python
MODEL_OPTIONS = [
    "vidore/colpali-v1.3",  # Primary choice
    "vidore/colpali",       # Fallback 1
    "vidore/colqwen2.5-v0.2",  # Fallback 2 (if available)
]

def load_colpali_model_with_fallback():
    for model_name in MODEL_OPTIONS:
        try:
            model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=device,
                cache_dir=volume_path,
                trust_remote_code=True,  # Critical for some models
                local_files_only=False
            )
            return model, processor, model_name
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            continue
```

#### 2. Update Transformers Version
Ensure you're using a compatible transformers version:

```bash
pip install --upgrade transformers>=4.51.0
```

#### 3. Clear Model Cache
Remove cached model files and retry:

```python
import shutil
import os

# Clear the cache directory
cache_dir = f"/Volumes/{catalog}/{schema}/{volume_label}"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared cache directory: {cache_dir}")
```

#### 4. Use Alternative Models
If ColPali continues to fail, consider these alternatives:

| Model | Pros | Cons | Memory Usage |
|-------|------|------|--------------|
| **CLIP** | Stable, fast | Lower recall | ~2GB |
| **ColQwen2** | Good performance | Larger model | ~8GB |
| **Custom fine-tuned** | Domain-specific | Requires training | Variable |

## Network and Connectivity Issues

### Symptoms
- Connection timeouts during model download
- Intermittent failures
- Slow download speeds

### Solutions

#### 1. Check Internet Connectivity
```python
import requests

def test_connectivity():
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        print(f"Connectivity test: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Connectivity test failed: {e}")
        return False
```

#### 2. Use Local Caching
Leverage Databricks volume storage for persistent caching:

```python
volume_path = f"/Volumes/{catalog}/{schema}/{volume_label}"
# Models will be cached here for faster subsequent loads
```

#### 3. Implement Retry Logic
```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
```

## Memory and Performance Issues

### Symptoms
- CUDA out of memory errors
- Slow inference times
- High RAM usage

### Solutions

#### 1. Use CPU Deployment
```python
device = torch.device("cpu")  # Stable for production
```

#### 2. Optimize Batch Processing
```python
def embed_pages_optimized(png_paths: List[str], batch_size: int = 4) -> List[torch.Tensor]:
    """Process images in smaller batches to reduce memory usage."""
    all_embeddings = []
    
    for i in range(0, len(png_paths), batch_size):
        batch_paths = png_paths[i:i + batch_size]
        imgs = [Image.open(p) for p in batch_paths]
        batch = processor.process_images(imgs).to(device)
        
        with torch.no_grad():
            embeddings = list(model(**batch))
            all_embeddings.extend(embeddings)
    
    return all_embeddings
```

#### 3. Model Quantization
```python
# For memory-constrained environments
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision
    device_map=device,
    cache_dir=volume_path,
    trust_remote_code=True
)
```

## Environment-Specific Issues

### Databricks Runtime Compatibility

#### 1. Check Runtime Version
Ensure you're using a compatible Databricks Runtime:

```python
import pyspark
print(f"Databricks Runtime: {pyspark.version}")
```

#### 2. Install Dependencies
```bash
%pip install --quiet --upgrade \
    "colpali-engine>=0.2.1" \
    "git+https://github.com/illuin-tech/colpali" \
    pillow-simd PyMuPDF \
    transformers>=4.51.0
```

### Unity Catalog Permissions

#### 1. Verify Volume Access
```python
# Test volume access
try:
    test_file = f"{volume_path}/test.txt"
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print("Volume access: OK")
except Exception as e:
    print(f"Volume access failed: {e}")
```

#### 2. Check Catalog Permissions
```sql
-- Verify catalog access
SHOW CATALOGS;
USE CATALOG your_catalog_name;
SHOW SCHEMAS;
```

## Performance Optimization

### 1. GPU Acceleration
```python
# Enable GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 2. PLAID Indexing
```python
def create_plaid_index(vectors: List[torch.Tensor]) -> Any:
    """Create PLAID index for faster retrieval."""
    # Implementation depends on colpali-engine version
    return processor.create_plaid_index(vectors)
```

### 3. Token Pooling
```python
# Reduce index size with minimal recall loss
def apply_token_pooling(embeddings: torch.Tensor, pool_factor: int = 3):
    """Apply token pooling to reduce index size."""
    # Implementation for token pooling
    pass
```

## Monitoring and Debugging

### 1. Model Loading Debug
```python
def debug_model_loading(model_name: str):
    """Debug model loading issues."""
    print(f"Attempting to load: {model_name}")
    
    # Check if model exists
    from huggingface_hub import model_info
    try:
        info = model_info(model_name)
        print(f"Model info: {info}")
    except Exception as e:
        print(f"Model info failed: {e}")
    
    # Check cache
    cache_dir = f"/Volumes/{catalog}/{schema}/{volume_label}"
    if os.path.exists(cache_dir):
        print(f"Cache contents: {os.listdir(cache_dir)}")
```

### 2. Performance Monitoring
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{operation_name} took {end_time - start_time:.2f} seconds")

# Usage
with timer("Model loading"):
    model, processor, model_name = load_colpali_model_with_fallback()
```

## Migration from CLIP

### Comparison Table
| Aspect | CLIP | ColPali |
|--------|------|---------|
| Embedding Type | Single-vector (1×768) | Multi-vector (~196×768) |
| Retrieval Method | Cosine similarity | Late-interaction |
| Spatial Information | Lost | Preserved |
| Recall Improvement | Baseline | +15-30pp |
| Index Size | Small | Large (~200x) |
| Query Speed | Fast | Slower but more accurate |

### Migration Checklist
- [ ] Convert PDF → PNG (already done for CLIP)
- [ ] Replace CLIP processor/model calls with ColPali equivalents
- [ ] Change index schema to multi-vector
- [ ] Update similarity_search code to late-interaction
- [ ] Retune ANN / pool-factor (optional)

## Getting Help

### 1. Check Official Resources
- [ColPali GitHub Repository](https://github.com/illuin-tech/colpali)
- [Hugging Face Model Page](https://huggingface.co/vidore/colpali-v1.3)
- [ColPali Paper](https://arxiv.org/abs/2407.01449)

### 2. Community Support
- [Hugging Face Discussions](https://huggingface.co/vidore/colpali-v1.3/discussions)
- [GitHub Issues](https://github.com/illuin-tech/colpali/issues)

### 3. Alternative Solutions
If ColPali continues to fail, consider:
1. **Stick with CLIP**: Use your existing implementation
2. **Try ColQwen2**: Alternative multi-vector model
3. **Custom fine-tuning**: Train on your specific domain

## Summary

The most common ColPali loading issues can be resolved by:
1. **Using the fallback mechanism** (implemented in the updated code)
2. **Updating transformers** to version 4.51+
3. **Adding trust_remote_code=True** to model loading
4. **Clearing corrupted cache** files
5. **Checking network connectivity**

The updated implementation includes robust error handling and fallback options to ensure successful deployment in production environments. 