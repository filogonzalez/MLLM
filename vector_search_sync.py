# Databricks notebook source
# Vector Search Sync - Automatic and manual syncing for new PDFs
# Handles incremental updates to vector search index

# MAGIC %pip install --upgrade databricks-vectorsearch databricks-sdk

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk.service.vectorsearch import EndpointType

# Import our global configuration
from global_config import get_config, get_full_table_name, get_full_index_name

# COMMAND ----------

# Initialize configuration
config = get_config()

# Get configuration values
endpoint_name = config["vector_search"]["endpoint_name"]
index_name = get_full_index_name()
embeddings_table = get_full_table_name("embeddings_table")
pdf_pages_table = get_full_table_name("pdf_pages_table")

print("Vector Search Sync Configuration:")
print(f"  Endpoint: {endpoint_name}")
print(f"  Index: {index_name}")
print(f"  Embeddings Table: {embeddings_table}")
print(f"  PDF Pages Table: {pdf_pages_table}")

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient(disable_notice=True)


def get_index_sync_status() -> Dict[str, Any]:
    """
    Get current sync status of the vector search index
    """
    try:
        # Get index details
        indexes = vsc.list_indexes(name=endpoint_name)
        
        # Find our specific index
        index_details = None
        for idx in indexes.get("vector_indexes", []):
            if idx.get("name") == index_name:
                index_details = idx
                break
        
        if not index_details:
            return {
                "status": "NOT_FOUND",
                "message": f"Index {index_name} not found",
                "last_sync": None,
                "sync_status": "UNKNOWN"
            }
        
        # Get sync status
        sync_status = index_details.get("sync_status", "UNKNOWN")
        last_sync = index_details.get("last_sync_time", None)
        
        return {
            "status": "FOUND",
            "message": f"Index {index_name} found",
            "last_sync": last_sync,
            "sync_status": sync_status,
            "index_details": index_details
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Error getting sync status: {e}",
            "last_sync": None,
            "sync_status": "ERROR"
        }


def trigger_index_sync(wait_for_completion: bool = True, 
                      timeout_minutes: int = 30) -> Dict[str, Any]:
    """
    Trigger a manual sync of the vector search index
    """
    try:
        print(f"Triggering sync for index: {index_name}")
        
        # Get the index
        index = vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        
        # Trigger sync
        sync_result = index.sync()
        print(f"Sync triggered: {sync_result}")
        
        if wait_for_completion:
            print("Waiting for sync to complete...")
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60
            
            while time.time() - start_time < timeout_seconds:
                status = get_index_sync_status()
                
                if status["sync_status"] == "SYNCED":
                    print("✓ Sync completed successfully!")
                    return {
                        "success": True,
                        "message": "Sync completed successfully",
                        "sync_status": status["sync_status"],
                        "last_sync": status["last_sync"]
                    }
                elif status["sync_status"] == "SYNCING":
                    print("⏳ Still syncing...")
                    time.sleep(30)  # Wait 30 seconds before checking again
                else:
                    print(f"⚠ Unexpected sync status: {status['sync_status']}")
                    time.sleep(10)
            
            print(f"⚠ Sync timeout after {timeout_minutes} minutes")
            return {
                "success": False,
                "message": f"Sync timeout after {timeout_minutes} minutes",
                "sync_status": "TIMEOUT"
            }
        else:
            return {
                "success": True,
                "message": "Sync triggered (not waiting for completion)",
                "sync_status": "TRIGGERED"
            }
            
    except Exception as e:
        print(f"Error triggering sync: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error triggering sync: {e}",
            "sync_status": "ERROR"
        }


def check_for_new_pdfs(since_timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if there are new PDFs that need processing
    """
    try:
        # If no timestamp provided, check last 24 hours
        if not since_timestamp:
            since_time = datetime.now() - timedelta(hours=24)
            since_timestamp = since_time.isoformat()
        
        print(f"Checking for new PDFs since: {since_timestamp}")
        
        # Query PDF pages table for recent entries
        query = f"""
        SELECT 
            pdf_path,
            page_number,
            processing_timestamp,
            COUNT(*) as page_count
        FROM {pdf_pages_table}
        WHERE processing_timestamp >= '{since_timestamp}'
        GROUP BY pdf_path, page_number, processing_timestamp
        ORDER BY processing_timestamp DESC
        """
        
        recent_pages = spark.sql(query)
        recent_count = recent_pages.count()
        
        if recent_count > 0:
            # Get unique PDFs
            unique_pdfs = recent_pages.select("pdf_path").distinct().collect()
            pdf_list = [row.pdf_path for row in unique_pdfs]
            
            print(f"Found {recent_count} new pages from {len(pdf_list)} PDFs")
            for pdf in pdf_list:
                print(f"  - {pdf}")
            
            return {
                "has_new_pdfs": True,
                "page_count": recent_count,
                "pdf_count": len(pdf_list),
                "pdfs": pdf_list,
                "since_timestamp": since_timestamp
            }
        else:
            print("No new PDFs found")
            return {
                "has_new_pdfs": False,
                "page_count": 0,
                "pdf_count": 0,
                "pdfs": [],
                "since_timestamp": since_timestamp
            }
            
    except Exception as e:
        print(f"Error checking for new PDFs: {e}")
        import traceback
        traceback.print_exc()
        return {
            "has_new_pdfs": False,
            "page_count": 0,
            "pdf_count": 0,
            "pdfs": [],
            "error": str(e)
        }


def check_embeddings_sync_status() -> Dict[str, Any]:
    """
    Check if embeddings table has new data that needs to be synced
    """
    try:
        # Check embeddings table for recent entries
        query = f"""
        SELECT 
            COUNT(*) as total_embeddings,
            MAX(processing_timestamp) as latest_embedding,
            COUNT(CASE WHEN processing_timestamp >= DATE_SUB(NOW(), 1) THEN 1 END) as recent_embeddings
        FROM {embeddings_table}
        """
        
        embedding_stats = spark.sql(query).collect()[0]
        
        # Check PDF pages table for comparison
        pdf_query = f"""
        SELECT 
            COUNT(*) as total_pages,
            MAX(processing_timestamp) as latest_page
        FROM {pdf_pages_table}
        """
        
        pdf_stats = spark.sql(pdf_query).collect()[0]
        
        # Check if there's a mismatch
        has_new_embeddings = (
            embedding_stats.recent_embeddings > 0 or
            embedding_stats.latest_embedding != pdf_stats.latest_page
        )
        
        return {
            "total_embeddings": embedding_stats.total_embeddings,
            "total_pages": pdf_stats.total_pages,
            "latest_embedding": embedding_stats.latest_embedding,
            "latest_page": pdf_stats.latest_page,
            "recent_embeddings": embedding_stats.recent_embeddings,
            "has_new_embeddings": has_new_embeddings,
            "needs_sync": has_new_embeddings
        }
        
    except Exception as e:
        print(f"Error checking embeddings sync status: {e}")
        return {
            "error": str(e),
            "needs_sync": False
        }


def auto_sync_pipeline() -> Dict[str, Any]:
    """
    Automated sync pipeline that checks for new PDFs and syncs if needed
    """
    print("\n" + "="*50)
    print("Auto Sync Pipeline")
    print("="*50)
    
    # Step 1: Check for new PDFs
    print("\n1. Checking for new PDFs...")
    pdf_check = check_for_new_pdfs()
    
    if not pdf_check["has_new_pdfs"]:
        print("No new PDFs found - no sync needed")
        return {
            "sync_triggered": False,
            "reason": "No new PDFs",
            "pdf_check": pdf_check
        }
    
    # Step 2: Check embeddings sync status
    print("\n2. Checking embeddings sync status...")
    embedding_check = check_embeddings_sync_status()
    
    if not embedding_check.get("needs_sync", False):
        print("Embeddings are up to date - no sync needed")
        return {
            "sync_triggered": False,
            "reason": "Embeddings up to date",
            "pdf_check": pdf_check,
            "embedding_check": embedding_check
        }
    
    # Step 3: Trigger sync
    print("\n3. Triggering vector search index sync...")
    sync_result = trigger_index_sync(wait_for_completion=True, timeout_minutes=30)
    
    return {
        "sync_triggered": True,
        "reason": "New PDFs detected",
        "pdf_check": pdf_check,
        "embedding_check": embedding_check,
        "sync_result": sync_result
    }


def manual_sync_with_options(
    force_sync: bool = False,
    wait_for_completion: bool = True,
    timeout_minutes: int = 30
) -> Dict[str, Any]:
    """
    Manual sync with various options
    """
    print("\n" + "="*50)
    print("Manual Sync")
    print("="*50)
    
    # Check current status
    print("\n1. Checking current index status...")
    status = get_index_sync_status()
    print(f"Index Status: {status['status']}")
    print(f"Sync Status: {status['sync_status']}")
    print(f"Last Sync: {status['last_sync']}")
    
    if not force_sync:
        # Check if sync is actually needed
        print("\n2. Checking if sync is needed...")
        pdf_check = check_for_new_pdfs()
        embedding_check = check_embeddings_sync_status()
        
        if not pdf_check["has_new_pdfs"] and not embedding_check.get("needs_sync", False):
            print("No sync needed - data is up to date")
            return {
                "sync_triggered": False,
                "reason": "No sync needed",
                "status": status,
                "pdf_check": pdf_check,
                "embedding_check": embedding_check
            }
    
    # Trigger sync
    print("\n3. Triggering sync...")
    sync_result = trigger_index_sync(
        wait_for_completion=wait_for_completion,
        timeout_minutes=timeout_minutes
    )
    
    return {
        "sync_triggered": True,
        "reason": "Manual sync triggered",
        "status": status,
        "sync_result": sync_result
    }


# COMMAND ----------

# Test the sync functionality
print("\n" + "="*50)
print("Testing Vector Search Sync")
print("="*50)

# Test 1: Check current status
print("\n--- Test 1: Current Status ---")
status = get_index_sync_status()
print(f"Status: {status}")

# Test 2: Check for new PDFs
print("\n--- Test 2: New PDFs Check ---")
pdf_check = check_for_new_pdfs()
print(f"PDF Check: {pdf_check}")

# Test 3: Check embeddings sync status
print("\n--- Test 3: Embeddings Sync Status ---")
embedding_check = check_embeddings_sync_status()
print(f"Embedding Check: {embedding_check}")

# Test 4: Auto sync pipeline (dry run)
print("\n--- Test 4: Auto Sync Pipeline (Dry Run) ---")
auto_result = auto_sync_pipeline()
print(f"Auto Sync Result: {auto_result}")

# COMMAND ----------

# Manual sync options
print("\n" + "="*50)
print("Manual Sync Options")
print("="*50)

print("""
Available sync functions:

1. get_index_sync_status() - Check current sync status
2. trigger_index_sync(wait_for_completion=True, timeout_minutes=30) - Trigger sync
3. check_for_new_pdfs(since_timestamp=None) - Check for new PDFs
4. check_embeddings_sync_status() - Check embeddings sync status
5. auto_sync_pipeline() - Automated sync pipeline
6. manual_sync_with_options(force_sync=False, wait_for_completion=True, timeout_minutes=30) - Manual sync

Examples:
- auto_sync_pipeline()  # Check and sync if needed
- manual_sync_with_options(force_sync=True)  # Force sync regardless
- trigger_index_sync(wait_for_completion=False)  # Trigger sync without waiting
""")

# COMMAND ----------

# Set up automatic sync (can be called from other notebooks)
def setup_automatic_sync():
    """
    Set up automatic sync that can be called from other pipelines
    """
    print("Setting up automatic sync...")
    
    # Check if sync is needed
    auto_result = auto_sync_pipeline()
    
    if auto_result["sync_triggered"]:
        print("✓ Automatic sync completed successfully")
        return True
    else:
        print("ℹ No sync needed at this time")
        return False


# Make functions available for import
__all__ = [
    "get_index_sync_status",
    "trigger_index_sync", 
    "check_for_new_pdfs",
    "check_embeddings_sync_status",
    "auto_sync_pipeline",
    "manual_sync_with_options",
    "setup_automatic_sync"
]

print("\n✓ Vector search sync module ready!")
print("Functions available for import and use in other notebooks.") 