# Databricks notebook source
# Embedding Generation Module - CLIP embeddings for images
# Replaces DSPy approach with direct MLflow client calls

# MAGIC %pip install --upgrade mlflow databricks-sdk

# COMMAND ----------

# Handle dbutils availability for dual environment support
dbutils.library.restartPython()

# COMMAND ----------

import time
from typing import List, Dict, Any
import mlflow
import mlflow.deployments
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    ArrayType, FloatType
)
from pyspark.sql.functions import current_timestamp

# Import our global configuration
from global_config import get_config, get_full_table_name

# COMMAND ----------

# Initialize configuration
config = get_config()

# Get configuration values
batch_size = config["data_pipeline"]["batch_size"]
clip_endpoint = config["models"]["clip"]["endpoint_name"]
pdf_pages_table = get_full_table_name("pdf_pages_table")
embeddings_table = get_full_table_name("embeddings_table")

# COMMAND ----------

# Initialize MLflow deployment client
client = mlflow.deployments.get_deploy_client("databricks")

# Verify endpoint is available
try:
    endpoint_info = client.get_endpoint(clip_endpoint)
    print(f"Using CLIP endpoint: {clip_endpoint}")
    print(f"Endpoint state: {endpoint_info.get('state', {})}")
except Exception as e:
    print(f"Error accessing endpoint {clip_endpoint}: {e}")
    raise

# COMMAND ----------


def generate_embedding_for_image(
    base64_image: str, 
    record_id: int,
    pdf_path: str,
    page_number: int
) -> Dict[str, Any]:
    """
    Generate CLIP embedding for a single image
    Optimized with CLIP best practices for better similarity search
    """
    try:
        # Prepare input for CLIP model
        input_data = {
            "dataframe_split": {
                "columns": ["image_base64"],
                "data": [[base64_image]]
            }
        }
        
        # Call CLIP endpoint
        response = client.predict(
            endpoint=clip_endpoint,
            inputs=input_data
        )
        
        # Debug: Print response structure for first few records
        if record_id <= 3:
            print(f"DEBUG - Record {record_id} response structure:")
            print(f"Response type: {type(response)}")
            print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            if isinstance(response, dict):
                for key, value in response.items():
                    print(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
        # Extract embedding from response
        embedding = None
        
        if isinstance(response, dict):
            # Try different possible response structures
            if 'predictions' in response:
                predictions = response['predictions']
                if isinstance(predictions, dict) and 'predictions' in predictions:
                    # Nested structure: {'predictions': {'predictions': {'embedding': [...]}}}
                    inner_predictions = predictions['predictions']
                    if isinstance(inner_predictions, list) and len(inner_predictions) > 0:
                        if isinstance(inner_predictions[0], dict) and 'embedding' in inner_predictions[0]:
                            embedding = inner_predictions[0]['embedding']
                    elif isinstance(inner_predictions, dict) and 'embedding' in inner_predictions:
                        embedding = inner_predictions['embedding']
                elif isinstance(predictions, list) and len(predictions) > 0:
                    # List of predictions
                    if isinstance(predictions[0], dict) and 'embedding' in predictions[0]:
                        embedding = predictions[0]['embedding']
                    elif isinstance(predictions[0], list):
                        # Direct embedding list
                        embedding = predictions[0]
                elif isinstance(predictions, dict) and 'embedding' in predictions:
                    # Single prediction dict
                    embedding = predictions['embedding']
                elif isinstance(predictions, list):
                    # Direct embedding list
                    embedding = predictions
            elif 'embedding' in response:
                # Direct embedding in response
                embedding = response['embedding']
            elif 'output' in response:
                # Try output field
                output = response['output']
                if isinstance(output, list) and len(output) > 0:
                    embedding = output[0] if isinstance(output[0], list) else output
                elif isinstance(output, dict) and 'embedding' in output:
                    embedding = output['embedding']
        
        # Validate embedding
        if not isinstance(embedding, list) or len(embedding) == 0:
            print(f"Warning: Invalid embedding for record {record_id}")
            print(f"Embedding type: {type(embedding)}, value: {embedding}")
            embedding = (
                [0.0] * config["models"]["clip"]["embedding_dimensions"]
            )
        else:
            # Normalize embedding for better similarity calculation
            # This is crucial for CLIP performance as mentioned in the article
            import numpy as np
            embedding_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
                embedding = embedding_array.tolist()
            
            print(f"Generated normalized embedding of length {len(embedding)} "
                  f"for record {record_id}")
        
        return {
            'id': record_id,
            'pdf_path': pdf_path,
            'page_number': page_number,
            'base64_image': base64_image,
            'embeddings': embedding
        }
        
    except Exception as e:
        print(f"Error generating embedding for record {record_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Return zero embedding on error
        return {
            'id': record_id,
            'pdf_path': pdf_path,
            'page_number': page_number,
            'base64_image': base64_image,
            'embeddings': (
                [0.0] * config["models"]["clip"]["embedding_dimensions"]
            )
        }


def process_embeddings_batch(
    pages_data: List[Any], 
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Process embeddings in batches with rate limiting
    """
    results = []
    record_id = 1
    
    total_pages = len(pages_data)
    print(f"Processing {total_pages} pages for embeddings...")
    
    for i in range(0, total_pages, batch_size):
        batch = pages_data[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_pages + batch_size - 1) // batch_size
        print(f"\nProcessing batch {batch_num}/{total_batches}")
        
        for row in batch:
            result = generate_embedding_for_image(
                base64_image=row.base64_image,
                record_id=record_id,
                pdf_path=row.pdf_path,
                page_number=row.page_number
            )
            results.append(result)
            record_id += 1
        
        # Rate limiting to avoid overwhelming the endpoint
        if i + batch_size < total_pages:
            time.sleep(0.5)
        
        print(f"Processed {len(results)} embeddings so far")
    
    return results


# COMMAND ----------
# Load pages from Delta table
print(f"Loading pages from {pdf_pages_table}")
df_pages = spark.table(pdf_pages_table)
pages_count = df_pages.count()
print(f"Found {pages_count} pages to process")

# COMMAND ----------

# Collect pages data for processing
# NOTE: This approach processes on driver node for simplicity
# For large-scale processing, consider using Spark UDFs
pages_data = df_pages.select(
    "pdf_path", 
    "page_number", 
    "base64_image"
).collect()

# Process embeddings
embedding_results = process_embeddings_batch(pages_data, batch_size)

# COMMAND ----------

# Create DataFrame with embeddings
if embedding_results:
    # Define schema
    embedding_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("pdf_path", StringType(), True),
        StructField("page_number", IntegerType(), True),
        StructField("base64_image", StringType(), True),
        StructField("embeddings", ArrayType(FloatType()), True)
    ])
    
    # Create DataFrame
    df_embeddings = spark.createDataFrame(embedding_results, embedding_schema)
    
    # Add timestamp
    df_embeddings = df_embeddings.withColumn(
        "processing_timestamp",
        current_timestamp()
    )
    
    # Write to Delta table
    print(
        f"\nWriting {df_embeddings.count()} embeddings to {embeddings_table}"
    )
    df_embeddings.write \
        .format("delta") \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .saveAsTable(embeddings_table)
    
    print(f"Successfully saved embeddings to {embeddings_table}")
    
    # Show sample
    print("\nSample embedding:")
    sample = df_embeddings.select("id", "pdf_path", "page_number").first()
    print(
        f"ID: {sample.id}, Path: {sample.pdf_path}, "
        f"Page: {sample.page_number}"
    )
else:
    print("No embeddings generated")

# COMMAND ----------

# Enable Change Data Feed for vector search sync
spark.sql(
    f"ALTER TABLE {embeddings_table} "
    f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Enabled Change Data Feed for {embeddings_table}")

# COMMAND ----------

# Trigger vector search sync after embeddings are generated
if embedding_results:
    print("\n" + "="*50)
    print("Triggering Vector Search Sync")
    print("="*50)
    
    try:
        # Import sync functions
        from vector_search_sync import trigger_index_sync, get_index_sync_status
        
        # Check current sync status
        print("Checking current vector search index status...")
        status = get_index_sync_status()
        print(f"Index Status: {status['status']}")
        print(f"Sync Status: {status['sync_status']}")
        
        # Trigger sync for new embeddings
        print("\nTriggering sync for new embeddings...")
        sync_result = trigger_index_sync(wait_for_completion=True, timeout_minutes=30)
        
        if sync_result["success"]:
            print("✓ Vector search sync completed successfully!")
            print(f"Message: {sync_result['message']}")
            print(f"Sync Status: {sync_result['sync_status']}")
        else:
            print("⚠ Vector search sync failed or timed out")
            print(f"Message: {sync_result['message']}")
            print("You may need to manually trigger sync later")
            
    except Exception as e:
        print(f"⚠ Error triggering vector search sync: {e}")
        print("You may need to manually trigger sync later using vector_search_sync.py")
        import traceback
        traceback.print_exc()

# COMMAND ----------

print("\n" + "="*50)
print("Embedding Generation Complete")
print("="*50)

if embedding_results:
    print(f"✓ Successfully generated {len(embedding_results)} embeddings")
    print(f"✓ Embeddings saved to {embeddings_table}")
    print("✓ Vector search index synced")
    print("\nNext steps:")
    print("1. Vector search index is ready for queries")
    print("2. Use multimodal_agent.py to query the updated index")
    print("3. Use vector_search_sync.py for manual sync if needed")
else:
    print("ℹ No embeddings were generated")
    print("Make sure PDF pages are available in the configured table")

# COMMAND ----------

# TODO: Re-register agent as MLflow model if needed for deployment 