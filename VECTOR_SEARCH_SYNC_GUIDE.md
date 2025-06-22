# Vector Search Sync Guide

## Overview

This guide explains how to sync the vector search index when new PDF files arrive in your multimodal RAG system. The sync process ensures that new embeddings are available for similarity search queries.

## Sync Process Flow

```
New PDFs → PDF Processing → Embedding Generation → Vector Search Sync → Ready for Queries
```

### 1. PDF Ingestion (`pdf_ingestion_pipeline.py`)
- Processes new PDF files into page images
- Saves to `pdf_pages_table`
- **Automatically triggers sync check**

### 2. Embedding Generation (`embedding_generation.py`)
- Generates CLIP embeddings for page images
- Saves to `embeddings_table`
- **Automatically triggers vector search sync**

### 3. Vector Search Sync (`vector_search_sync.py`)
- Syncs embeddings table to vector search index
- Makes new embeddings available for queries

## Automatic Sync

### Built-in Automatic Sync

The system automatically triggers sync in two scenarios:

1. **After PDF Processing**: When new PDFs are processed in `pdf_ingestion_pipeline.py`
2. **After Embedding Generation**: When new embeddings are generated in `embedding_generation.py`

### How Automatic Sync Works

```python
# Automatic sync is triggered after processing
if all_page_data:
    from vector_search_sync import auto_sync_pipeline
    sync_result = auto_sync_pipeline()
    
    if sync_result["sync_triggered"]:
        print("✓ Vector search sync completed successfully!")
```

### Automatic Sync Pipeline

The `auto_sync_pipeline()` function:

1. **Checks for new PDFs** in the last 24 hours
2. **Checks embeddings sync status** to see if new embeddings exist
3. **Triggers sync** only if needed
4. **Waits for completion** and reports status

## Manual Sync Options

### 1. Check Sync Status

```python
from vector_search_sync import get_index_sync_status

status = get_index_sync_status()
print(f"Index Status: {status['status']}")
print(f"Sync Status: {status['sync_status']}")
print(f"Last Sync: {status['last_sync']}")
```

### 2. Trigger Manual Sync

```python
from vector_search_sync import trigger_index_sync

# Sync and wait for completion
sync_result = trigger_index_sync(wait_for_completion=True, timeout_minutes=30)

# Sync without waiting
sync_result = trigger_index_sync(wait_for_completion=False)
```

### 3. Check for New PDFs

```python
from vector_search_sync import check_for_new_pdfs

# Check last 24 hours
pdf_check = check_for_new_pdfs()

# Check since specific timestamp
pdf_check = check_for_new_pdfs(since_timestamp="2024-01-01T00:00:00")
```

### 4. Check Embeddings Sync Status

```python
from vector_search_sync import check_embeddings_sync_status

embedding_check = check_embeddings_sync_status()
print(f"Needs sync: {embedding_check['needs_sync']}")
print(f"Total embeddings: {embedding_check['total_embeddings']}")
```

### 5. Manual Sync with Options

```python
from vector_search_sync import manual_sync_with_options

# Smart sync (only if needed)
result = manual_sync_with_options(force_sync=False)

# Force sync regardless
result = manual_sync_with_options(force_sync=True)

# Sync without waiting
result = manual_sync_with_options(wait_for_completion=False)
```

## Sync Functions Reference

### `get_index_sync_status()`
Returns current sync status of the vector search index.

**Returns:**
```python
{
    "status": "FOUND",  # FOUND, NOT_FOUND, ERROR
    "message": "Index found",
    "last_sync": "2024-01-01T12:00:00",
    "sync_status": "SYNCED"  # SYNCED, SYNCING, UNKNOWN, ERROR
}
```

### `trigger_index_sync(wait_for_completion=True, timeout_minutes=30)`
Triggers a manual sync of the vector search index.

**Parameters:**
- `wait_for_completion`: Whether to wait for sync to complete
- `timeout_minutes`: Maximum time to wait for completion

**Returns:**
```python
{
    "success": True,
    "message": "Sync completed successfully",
    "sync_status": "SYNCED"
}
```

### `check_for_new_pdfs(since_timestamp=None)`
Checks if there are new PDFs that need processing.

**Parameters:**
- `since_timestamp`: ISO timestamp to check from (default: 24 hours ago)

**Returns:**
```python
{
    "has_new_pdfs": True,
    "page_count": 10,
    "pdf_count": 2,
    "pdfs": ["/path/to/doc1.pdf", "/path/to/doc2.pdf"]
}
```

### `check_embeddings_sync_status()`
Checks if embeddings table has new data that needs to be synced.

**Returns:**
```python
{
    "total_embeddings": 100,
    "total_pages": 100,
    "latest_embedding": "2024-01-01T12:00:00",
    "latest_page": "2024-01-01T12:00:00",
    "recent_embeddings": 5,
    "has_new_embeddings": True,
    "needs_sync": True
}
```

### `auto_sync_pipeline()`
Automated sync pipeline that checks for new PDFs and syncs if needed.

**Returns:**
```python
{
    "sync_triggered": True,
    "reason": "New PDFs detected",
    "pdf_check": {...},
    "embedding_check": {...},
    "sync_result": {...}
}
```

### `manual_sync_with_options(force_sync=False, wait_for_completion=True, timeout_minutes=30)`
Manual sync with various options.

**Parameters:**
- `force_sync`: Force sync regardless of whether it's needed
- `wait_for_completion`: Whether to wait for sync to complete
- `timeout_minutes`: Maximum time to wait for completion

## Sync Status Values

### Index Status
- `FOUND`: Index exists and is accessible
- `NOT_FOUND`: Index doesn't exist
- `ERROR`: Error accessing index

### Sync Status
- `SYNCED`: Index is up to date
- `SYNCING`: Index is currently syncing
- `UNKNOWN`: Sync status cannot be determined
- `ERROR`: Error occurred during sync
- `TIMEOUT`: Sync timed out

## Common Sync Scenarios

### Scenario 1: New PDFs Added
```python
# 1. Add PDFs to volume
# 2. Run pdf_ingestion_pipeline.py (automatic sync triggered)
# 3. Run embedding_generation.py (automatic sync triggered)
# 4. Index is ready for queries
```

### Scenario 2: Manual Sync Needed
```python
from vector_search_sync import auto_sync_pipeline

# Check and sync if needed
result = auto_sync_pipeline()
if result["sync_triggered"]:
    print("✓ Sync completed")
else:
    print("ℹ No sync needed")
```

### Scenario 3: Force Sync
```python
from vector_search_sync import manual_sync_with_options

# Force sync regardless of status
result = manual_sync_with_options(force_sync=True)
```

### Scenario 4: Check Sync Health
```python
from vector_search_sync import get_index_sync_status, check_embeddings_sync_status

# Check index status
status = get_index_sync_status()
print(f"Index: {status['status']}, Sync: {status['sync_status']}")

# Check embeddings status
embedding_status = check_embeddings_sync_status()
print(f"Needs sync: {embedding_status['needs_sync']}")
```

## Troubleshooting

### Sync Not Triggering
1. Check if new PDFs were actually processed
2. Verify embeddings were generated
3. Check Change Data Feed is enabled on embeddings table

### Sync Failing
1. Check vector search endpoint status
2. Verify index exists and is accessible
3. Check for sufficient permissions
4. Review error logs for specific issues

### Sync Taking Too Long
1. Check index size and complexity
2. Consider increasing timeout
3. Monitor cluster resources
4. Check for network issues

### Manual Sync Commands
```python
# Quick status check
from vector_search_sync import get_index_sync_status
get_index_sync_status()

# Force sync
from vector_search_sync import trigger_index_sync
trigger_index_sync(wait_for_completion=True, timeout_minutes=60)

# Check what needs syncing
from vector_search_sync import check_for_new_pdfs, check_embeddings_sync_status
check_for_new_pdfs()
check_embeddings_sync_status()
```

## Best Practices

1. **Use Automatic Sync**: Let the pipeline handle sync automatically
2. **Monitor Sync Status**: Check sync status after major updates
3. **Set Appropriate Timeouts**: Use longer timeouts for large datasets
4. **Handle Errors Gracefully**: Always check sync results
5. **Log Sync Events**: Track sync history for debugging

## Integration with DLT Pipelines

For production use with DLT pipelines, you can integrate sync functions:

```python
# In DLT pipeline
@dlt.table
def sync_vector_search():
    from vector_search_sync import auto_sync_pipeline
    result = auto_sync_pipeline()
    return spark.createDataFrame([result])
```

This ensures vector search stays in sync with your data pipeline automatically. 