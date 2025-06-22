# MAGIC %md
# MAGIC # Example DLT Pipeline Using Bundle Variables
# MAGIC This notebook demonstrates how to use the global configuration with Databricks Asset Bundle variables

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from global_config import get_config, get_full_table_name

# Get configuration
config = get_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Layer - Raw PDF Pages

# COMMAND ----------

@dlt.table(
    name=config["tables"]["pdf_pages_table"],
    comment="Bronze layer: Raw PDF pages extracted from documents"
)
def bronze_pdf_pages():
    """
    Read raw PDF pages from the configured volume path
    """
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.pdf")
        .load(config["volumes"]["pdf_volume_path"])
        .select(
            F.col("path"),
            F.col("modificationTime"),
            F.col("length").alias("file_size"),
            F.current_timestamp().alias("ingestion_timestamp")
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer - Processed Images with Embeddings

# COMMAND ----------

@dlt.table(
    name=config["tables"]["embeddings_table"],
    comment="Silver layer: Processed images with CLIP embeddings",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_embedding_dimension", 
                    F.size("embedding") == config["models"]["clip"]["embedding_dimensions"])
@dlt.expect_or_drop("valid_image_format", 
                    F.col("image_format").isin(config["data_pipeline"]["image_processing"]["supported_formats"]))
def silver_embeddings():
    """
    Process images and generate CLIP embeddings
    """
    return (
        dlt.read_stream("bronze_pdf_pages")
        .select(
            F.col("path"),
            F.col("ingestion_timestamp"),
            # Placeholder for actual image processing and embedding generation
            F.lit("JPEG").alias("image_format"),
            F.array([F.lit(0.0) for _ in range(int(config["models"]["clip"]["embedding_dimensions"]))]).alias("embedding"),
            F.current_timestamp().alias("processing_timestamp")
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Layer - Vector Search Index

# COMMAND ----------

@dlt.table(
    name=config["tables"]["vector_index_table"],
    comment="Gold layer: Vector search index for similarity queries",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true",
        "delta.enableChangeDataFeed": "true"
    }
)
def gold_vector_index():
    """
    Prepare data for vector search index
    """
    return (
        dlt.read("silver_embeddings")
        .select(
            F.col("path").alias("document_path"),
            F.col("embedding"),
            F.col("image_format"),
            F.col("processing_timestamp"),
            # Add metadata for search
            F.struct(
                F.col("path").alias("source_path"),
                F.col("image_format"),
                F.col("processing_timestamp")
            ).alias("metadata")
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Summary

# COMMAND ----------

# Display current configuration for verification
if spark.conf.get("pipelines.id", None) is None:
    # Only display when not running in DLT pipeline
    import json
    print("Current Configuration:")
    print(json.dumps(config, indent=2))
    
    print("\nFully Qualified Table Names:")
    print(f"PDF Pages: {get_full_table_name('pdf_pages_table')}")
    print(f"Embeddings: {get_full_table_name('embeddings_table')}")
    print(f"Vector Index: {get_full_table_name('vector_index_table')}") 