# Global Configuration for Multimodal RAG Pipeline
# This configuration integrates with Databricks Asset Bundle variables

import os
from bundle_variables import get_all_variables

# Get all bundle variables
bundle_variables = get_all_variables()

# Base configuration that references bundle variables
# In Databricks, these will be resolved from bundle.variables.yml
base_config = {
    # Environment flags
    "environment": {
        "is_databricks": True,  # Set to False for local development
    },
    
    # Unity Catalog Configuration
    "unity_catalog": {
        "catalog": bundle_variables.get("catalog", ""),
        "schema": bundle_variables.get("schema", ""),
        "rag_app_name": bundle_variables.get("rag_app_name", ""),
    },
    
    # Volume Configuration
    "volumes": {
        "volume_label": bundle_variables.get("volume_label", ""),
        "volume_path": bundle_variables.get("volume_path", ""),
        "pdf_volume_path": bundle_variables.get("pdf_volume_path", ""),
    },
    
    # Model Configuration
    "models": {
        # CLIP Model Configuration
        "clip": {
            "model_name": "clip_model_embedder",
            "endpoint_name": bundle_variables.get("clip_endpoint_name", ""),
            "registered_model_name": (
                bundle_variables.get("clip_model_registered_name", "")
            ),
            "clip_model_name": bundle_variables.get("clip_model_name", ""),
            "embedding_dimensions": (
                bundle_variables.get("clip_embedding_dimensions", 512)
            ),
            "model_config": {
                "trust_remote_code": True,
                "force_download": False,
            }
        },
        
        # LLM Configuration (GPT-4o via Azure OpenAI)
        "llm": {
            "endpoint_name": bundle_variables.get("llm_endpoint_name", ""),
            "llm_parameters": {
                "temperature": bundle_variables.get("llm_temperature", 0),
                "max_tokens": bundle_variables.get("llm_max_tokens", 2000)
            },
            "llm_system_prompt_template": (
                "You are an insightful and helpful assistant for BCP that "
                "only answers questions related to BCP internal "
                "documentation. Use the following pieces of retrieved "
                "context to answer the question. Some pieces of context "
                "may be irrelevant, in which case you should not use them "
                "to form the answer. Answer honestly and if you do not "
                "know the answer or if the answer is not contained in the "
                "documentation provided as context, limit yourself to "
                "answer that 'You could not find the answer in the "
                "documentation and prompt the user to provide more "
                "details'\n\n"
                "For each piece of information you use from the context, "
                "you MUST include the source document path at the end of "
                "your response in the following format:\n"
                "Sources: - [exact_path_from_path]\n"
                "Do not modify or summarize the paths - use them exactly "
                "as provided in the path field.\n\n"
                "Context: {context}"
            ),
        }
    },
    
    # Vector Search Configuration
    "vector_search": {
        "endpoint_name": (
            bundle_variables.get("vector_search_endpoint_name", "")
        ),
        "index_name": bundle_variables.get("vector_search_index_name", ""),
        "table_name": bundle_variables.get("vector_search_table_name", ""),
        "pipeline_type": "TRIGGERED",
        "parameters": {
            "k": bundle_variables.get("vector_search_k", 5),
            "query_type": (
                bundle_variables.get("vector_search_query_type", "HYBRID")
            ),
        }
    },
    
    # Data Pipeline Configuration
    "data_pipeline": {
        "batch_size": bundle_variables.get("batch_size", 10),
        "image_processing": {
            "dpi": bundle_variables.get("image_dpi", 100),
            "format": bundle_variables.get("image_format", "JPEG"),
            "quality": bundle_variables.get("image_quality", 70),
            "max_dimension": (
                bundle_variables.get("image_max_dimension", 1568)
            ),
            "max_megapixels": (
                bundle_variables.get("image_max_megapixels", 1150000)
            ),
            "supported_formats": ["PNG", "JPEG"],
            "normalize": True
        },
        "chunk_size": 512,  # Text chunking (deprecated - images only)
        "chunk_overlap": 128,  # Text chunking (deprecated - images only)
    },
    
    # Table Names Configuration
    "tables": {
        "pdf_pages_table": bundle_variables.get("pdf_pages_table", ""),
        "embeddings_table": bundle_variables.get("embeddings_table", ""),
        "vector_index_table": bundle_variables.get("vector_index_table", ""),
    },
    
    # Azure OpenAI Configuration (for GPT-4o)
    "azure_openai": {
        "endpoint": bundle_variables.get("azure_endpoint", "https://fgbcpai.openai.azure.com/"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
        # Store in environment variable
        "api_version": bundle_variables.get("azure_api_version", "2025-01-01-preview"),
        "deployment_name": bundle_variables.get("azure_deployment_name", "gbcpt-4o"),
    }
}


def get_config():
    """Get the global configuration dictionary"""
    return base_config


def get_full_table_name(table_key):
    """Get fully qualified table name"""
    config = get_config()
    catalog = config["unity_catalog"]["catalog"]
    schema = config["unity_catalog"]["schema"]
    table_name = config["tables"][table_key]
    return f"{catalog}.{schema}.{table_name}"


def get_full_index_name():
    """Get fully qualified vector search index name"""
    config = get_config()
    catalog = config["unity_catalog"]["catalog"]
    schema = config["unity_catalog"]["schema"]
    index_name = config["vector_search"]["index_name"]
    return f"{catalog}.{schema}.{index_name}"


# COMMAND ----------

# Display configuration for verification
if __name__ == "__main__":
    import json
    print("Global Configuration:")
    print(json.dumps(get_config(), indent=2)) 
