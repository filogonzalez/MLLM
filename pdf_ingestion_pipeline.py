# Databricks notebook source
# PDF Ingestion Pipeline - Image-based processing only
# No text extraction - using multimodal approach with images

# MAGIC %pip install --upgrade pdf2image pillow mlflow databricks-sdk PyMuPDF popple

# COMMAND ----------

# Handle dbutils availability for dual environment support
dbutils.library.restartPython()

# COMMAND ----------

import os
import base64
import io
from pdf2image import convert_from_path
from PIL import Image
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)
from pyspark.sql.functions import current_timestamp

# Import our global configuration
from global_config import get_config, get_full_table_name

# COMMAND ----------

# Initialize configuration
config = get_config()

# Get configuration values
pdf_volume_path = config["volumes"]["pdf_volume_path"]
image_config = config["data_pipeline"]["image_processing"]
batch_size = config["data_pipeline"]["batch_size"]
pdf_pages_table = get_full_table_name("pdf_pages_table")

# COMMAND ----------
def install_poppler_on_nodes():
    """
    Install poppler on all cluster nodes
    """
    import subprocess
    import os

    try:
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'poppler-utils'], check=True)
        print("Poppler installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing proppler: {e}")

# Install poppler on all nodes
sc.range(1).foreach(lambda x: install_poppler_on_nodes())

# Also install poppler directly on driver node
try:
    import subprocess
    subprocess.run(['apt-get', 'update'], check=True)
    subprocess.run(['apt-get', 'install', '-y', 'poppler-utils'], check=True)
    print("Poppler installed on driver node")
except Exception as e:
    print(f"Could not install poppler on driver node: {e}")

# COMMAND ----------
def resize_image(img, max_px=1568, max_mp=1150000):
    """
    Resize image while maintaining aspect ratio
    """
    w, h = img.size
    if w * h > max_mp or max(w, h) > max_px:
        scale = min(max_px / max(w, h), (max_mp / (w * h)) ** 0.5)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img


def process_pdf_with_pymupdf(pdf_path):
    """
    Fallback PDF processing using PyMuPDF (fitz)
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            images.append(img)
        
        doc.close()
        return images
        
    except ImportError:
        print("PyMuPDF not available for fallback processing")
        return None
    except Exception as e:
        print(f"Error processing PDF with PyMuPDF: {e}")
        return None


def process_all_pdfs(pdf_paths):
    """
    Process all PDFs on driver node to avoid UDF distribution issues
    """
    all_pages = []
    
    for pdf_path in pdf_paths:
        try:
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                continue
            
            if not os.access(pdf_path, os.R_OK):
                print(f"File not readable: {pdf_path}")
                continue
            
            print(f"Processing: {pdf_path}")
            
            # Try pdf2image first
            try:
                images = convert_from_path(
                    pdf_path, 
                    dpi=100,
                    fmt='JPEG',
                    poppler_path='/usr/bin'  
                )
                print(f"Successfully processed with pdf2image")
            except Exception as e:
                print(f"pdf2image failed: {e}")
                print("Trying PyMuPDF fallback...")
                
                # Fallback to PyMuPDF
                images = process_pdf_with_pymupdf(pdf_path)
                if images is None:
                    print(f"Both pdf2image and PyMuPDF failed for {pdf_path}")
                    continue
                print(f"Successfully processed with PyMuPDF")
            
            for i, image in enumerate(images):
                resized_image = resize_image(image)
                
                if resized_image.mode != 'RGB':
                    resized_image = resized_image.convert('RGB')
                
                quantized_image = resized_image.quantize(colors=256)
                quantized_image = quantized_image.convert('RGB')
                
                img_buffer = io.BytesIO()
                quantized_image.save(img_buffer, format='JPEG', quality=70, optimize=True)
                img_bytes = img_buffer.getvalue()
                
                base64_string = base64.b64encode(img_bytes).decode('utf-8')
                
                # Create one record per page (no text chunks)
                all_pages.append({
                    'pdf_path': pdf_path,
                    'page_number': i + 1,
                    'base64_image': base64_string,
                })
            
            print(f"Successfully processed {len(images)} pages from {pdf_path}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_pages

# COMMAND ----------
# Get list of PDF files
pdf_files = []
if os.path.exists(pdf_volume_path):
    files = os.listdir(pdf_volume_path)
    pdf_files = [
        os.path.join(pdf_volume_path, f) 
        for f in files 
        if f.endswith('.pdf')
    ]
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f" - {pdf}")
else:
    print(f"PDF volume path not found: {pdf_volume_path}")

# COMMAND ----------

# Process all PDFs
print(f"Processing {len(pdf_files)} PDFs...")
all_page_data = process_all_pdfs(pdf_files)
print(f"Total pages processed: {len(all_page_data)}")

# COMMAND ----------
# Create DataFrame and save to Delta table
if all_page_data:
    # Define schema (no text_chunk field - images only)
    pdf_schema = StructType([
        StructField("pdf_path", StringType(), True),
        StructField("page_number", IntegerType(), True),
        StructField("base64_image", StringType(), True),
    ])
    
    # Create DataFrame
    df_pages = spark.createDataFrame(all_page_data, pdf_schema)
    
    # Add processing timestamp
    df_pages = df_pages.withColumn(
        "processing_timestamp", 
        current_timestamp()
    )
    
    # Write to Delta table
    print(f"Writing {df_pages.count()} pages to {pdf_pages_table}")
    df_pages.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(pdf_pages_table)
    
    print(f"Successfully saved pages to {pdf_pages_table}")
else:
    print("No pages to process")

# COMMAND ----------
# Display sample of processed data
if all_page_data:
    print("\nSample of processed data:")
    print(f"First page info: {all_page_data[0]['pdf_path']}, "
          f"Page {all_page_data[0]['page_number']}")
    print(f"Base64 image length: {len(all_page_data[0]['base64_image'])}")

# COMMAND ----------

# Trigger vector search sync if new PDFs were processed
if all_page_data:
    print("\n" + "="*50)
    print("Triggering Vector Search Sync")
    print("="*50)
    
    try:
        # Import sync functions
        from vector_search_sync import auto_sync_pipeline, get_index_sync_status
        
        # Check current sync status
        print("Checking current vector search index status...")
        status = get_index_sync_status()
        print(f"Index Status: {status['status']}")
        print(f"Sync Status: {status['sync_status']}")
        
        # Trigger automatic sync
        print("\nTriggering automatic sync for new PDFs...")
        sync_result = auto_sync_pipeline()
        
        if sync_result["sync_triggered"]:
            print("✓ Vector search sync completed successfully!")
            print(f"Reason: {sync_result['reason']}")
            if "sync_result" in sync_result:
                print(f"Sync Details: {sync_result['sync_result']}")
        else:
            print("ℹ No sync needed - embeddings may already be up to date")
            print(f"Reason: {sync_result['reason']}")
            
    except Exception as e:
        print(f"⚠ Error triggering vector search sync: {e}")
        print("You may need to manually trigger sync later using vector_search_sync.py")
        import traceback
        traceback.print_exc()

# COMMAND ----------

print("\n" + "="*50)
print("PDF Ingestion Pipeline Complete")
print("="*50)

if all_page_data:
    print(f"✓ Successfully processed {len(all_page_data)} pages")
    print(f"✓ Data saved to {pdf_pages_table}")
    print("✓ Vector search sync triggered")
    print("\nNext steps:")
    print("1. Run embedding_generation.py to generate CLIP embeddings")
    print("2. Vector search index will sync automatically")
    print("3. Use multimodal_agent.py to query the updated index")
else:
    print("ℹ No PDFs were processed")
    print("Make sure PDF files are available in the configured volume path")

# COMMAND ----------

# TODO: Delete this file once new pipeline is confirmed working
# This replaces the legacy streaming_document_pipeline.py 