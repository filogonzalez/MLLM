# Databricks notebook source
# Vector Search Setup - Create endpoints and indexes for image embeddings
# Optimized with CLIP best practices for better similarity search

# MAGIC %pip install --upgrade databricks-vectorsearch databricks-sdk

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk.service.vectorsearch import EndpointType
import mlflow.deployments
import numpy as np
from typing import List, Dict, Any

# Import our global configuration
from global_config import get_config, get_full_table_name, get_full_index_name

# COMMAND ----------

# Initialize configuration
config = get_config()

# Get configuration values
endpoint_name = config["vector_search"]["endpoint_name"]
index_name = get_full_index_name()
embeddings_table = get_full_table_name("embeddings_table")
pipeline_type = config["vector_search"]["pipeline_type"]
embedding_dimensions = config["models"]["clip"]["embedding_dimensions"]
clip_endpoint = config["models"]["clip"]["endpoint_name"]

print("Vector Search Configuration:")
print(f"  Endpoint: {endpoint_name}")
print(f"  Index: {index_name}")
print(f"  Source Table: {embeddings_table}")
print(f"  Pipeline Type: {pipeline_type}")
print(f"  CLIP Endpoint: {clip_endpoint}")

# COMMAND ----------

# Initialize Vector Search client and MLflow client
vsc = VectorSearchClient(disable_notice=True)
mlflow_client = mlflow.deployments.get_deploy_client("databricks")


def preprocess_query_for_clip(query: str) -> str:
    """
    Preprocess text query for better CLIP performance
    Based on CLIP best practices from the article
    """
    # Remove special characters and normalize
    query = query.strip().lower()
    
    # Add descriptive prefixes for better CLIP understanding
    # CLIP was trained on image-text pairs, so being descriptive helps
    prefixes = ("a photo of", "an image of", "a picture of", 
               "a document showing", "a page containing")
    if not query.startswith(prefixes):
        # For document queries, use document-specific prefixes
        doc_keywords = ["document", "pdf", "page", "form", "certificate", "cdv"]
        if any(word in query.lower() for word in doc_keywords):
            query = f"a document page showing {query}"
        else:
            query = f"a photo of {query}"
    
    print(f"Preprocessed query: '{query}'")
    return query


def generate_clip_embedding(text: str) -> List[float]:
    """
    Generate CLIP embedding for text query
    Optimized based on CLIP best practices
    """
    try:
        # Preprocess the query
        processed_text = preprocess_query_for_clip(text)
        
        # Generate embedding using CLIP endpoint
        response = mlflow_client.predict(
            endpoint=clip_endpoint,
            inputs={"dataframe_split": {
                "columns": ["text"],
                "data": [[processed_text]]
            }}
        )
        
        # Extract embedding from response
        embedding = None
        
        if isinstance(response, dict):
            if 'predictions' in response:
                predictions = response['predictions']
                if isinstance(predictions, dict) and 'predictions' in predictions:
                    inner_predictions = predictions['predictions']
                    if isinstance(inner_predictions, list) and len(inner_predictions) > 0:
                        if isinstance(inner_predictions[0], dict) and 'embedding' in inner_predictions[0]:
                            embedding = inner_predictions[0]['embedding']
                    elif isinstance(inner_predictions, dict) and 'embedding' in inner_predictions:
                        embedding = inner_predictions['embedding']
                elif isinstance(predictions, list) and len(predictions) > 0:
                    if isinstance(predictions[0], dict) and 'embedding' in predictions[0]:
                        embedding = predictions[0]['embedding']
                    elif isinstance(predictions[0], list):
                        embedding = predictions[0]
                elif isinstance(predictions, dict) and 'embedding' in predictions:
                    embedding = predictions['embedding']
                elif isinstance(predictions, list):
                    embedding = predictions
            elif 'embedding' in response:
                embedding = response['embedding']
        
        if not isinstance(embedding, list) or len(embedding) == 0:
            raise ValueError(f"Invalid embedding generated: {embedding}")
        
        # Normalize embedding for better similarity calculation
        embedding_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        print(f"Generated normalized embedding of length: {len(embedding_array)}")
        return embedding_array.tolist()
        
    except Exception as e:
        print(f"Error generating CLIP embedding: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    Based on CLIP's similarity calculation method
    """
    try:
        vec1_array = np.array(vec1, dtype=np.float32)
        vec2_array = np.array(vec2, dtype=np.float32)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_normalized = vec1_array / norm1
        vec2_normalized = vec2_array / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_normalized, vec2_normalized)
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0


def create_vector_search_endpoint():
    """Create the Vector Search endpoint if it doesn't exist"""
    try:
        # Check if endpoint exists
        endpoints = vsc.list_endpoints()
        endpoint_exists = any(
            endpoint["name"] == endpoint_name 
            for endpoint in endpoints.get("endpoints", [])
        )
        
        if not endpoint_exists:
            print(f"Creating Vector Search endpoint: {endpoint_name}")
            vsc.create_endpoint_and_wait(
                endpoint_name,
                endpoint_type=EndpointType.STANDARD.value
            )
            print("Endpoint created successfully!")
        else:
            print(f"Endpoint {endpoint_name} already exists.")
        
        print(f"PASS: Vector Search endpoint `{endpoint_name}` exists")
        return True
        
    except Exception as e:
        print(f"Error creating endpoint: {e}")
        return False


def create_vector_search_index():
    """Create the Vector Search index if it doesn't exist"""
    try:
        # Check if index exists
        indexes = vsc.list_indexes(name=endpoint_name)
        index_exists = any(
            index["name"] == index_name
            for index in indexes.get("vector_indexes", [])
        )
        
        if not index_exists:
            print(f"Creating Vector Search Index: {index_name}")
            print("This can take 15 minutes or longer...")
            
            # Create index with optimized settings for CLIP embeddings
            vsc.create_delta_sync_index_and_wait(
                endpoint_name=endpoint_name,
                index_name=index_name,
                primary_key="id",
                source_table_name=embeddings_table,
                pipeline_type=pipeline_type,
                embedding_vector_column="embeddings",
                embedding_dimension=embedding_dimensions
            )
            print("Index created successfully!")
        else:
            print(f"Index {index_name} already exists.")
            
        return True
        
    except Exception as e:
        print(f"Error creating index: {e}")
        import traceback
        traceback.print_exc()
        return False


def perform_optimized_similarity_search(
    query: str, 
    num_results: int = 10,
    similarity_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform optimized similarity search using CLIP best practices
    """
    try:
        # Get the index
        index = vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        
        # Generate optimized CLIP embedding
        query_embedding = generate_clip_embedding(query)
        
        # Perform similarity search with optimized parameters
        results = index.similarity_search(
            num_results=num_results * 2,  # Get more results for filtering
            columns=["base64_image", "pdf_path", "page_number", 
                    "processing_timestamp"],
            query_vector=query_embedding,
            query_text=query,
            query_type="HYBRID"
        )
        
        # Process and filter results
        processed_results = []
        
        if isinstance(results, list):
            result_data = results
        else:
            result_data = results.get('result', {}).get('data_array', [])
        
        for i, result in enumerate(result_data):
            # Handle both list and dict formats
            if isinstance(result, list):
                base64_image = result[0] if len(result) > 0 else ''
                pdf_path = result[1] if len(result) > 1 else 'Unknown'
                page_number = result[2] if len(result) > 2 else 'Unknown'
                processing_timestamp = result[3] if len(result) > 3 else 'Unknown'
                score = result[4] if len(result) > 4 else 0.0
            else:
                pdf_path = result.get('pdf_path', 'Unknown')
                page_number = result.get('page_number', 'Unknown')
                base64_image = result.get('base64_image', '')
                score = result.get('score', 0.0)
                processing_timestamp = result.get('processing_timestamp', 
                                                'Unknown')
            
            # Filter by similarity threshold
            if score >= similarity_threshold:
                processed_results.append({
                    'pdf_path': pdf_path,
                    'page_number': page_number,
                    'base64_image': base64_image,
                    'processing_timestamp': processing_timestamp,
                    'similarity_score': score
                })
        
        # Sort by similarity score and limit results
        processed_results.sort(key=lambda x: x['similarity_score'], 
                             reverse=True)
        processed_results = processed_results[:num_results]
        
        return processed_results
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        import traceback
        traceback.print_exc()
        return []


# COMMAND ----------

# Create endpoint
if create_vector_search_endpoint():
    print("\n✓ Vector Search endpoint ready")
else:
    print("\n✗ Failed to create endpoint")
    raise Exception("Cannot proceed without vector search endpoint")

# COMMAND ----------

# Create index
if create_vector_search_index():
    print("\n✓ Vector Search index ready")
    
    # Get index info
    try:
        index_obj = vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        print(f"\nIndex created successfully: {index_name}")
        print(f"Index object type: {type(index_obj)}")
    except Exception as e:
        print(f"Could not get index info: {e}")
else:
    print("\n✗ Failed to create index")
    raise Exception("Cannot proceed without vector search index")

# COMMAND ----------

# Test the optimized index with multiple queries
print("\n" + "="*50)
print("Testing Optimized Vector Search Index")
print("="*50)

# Test queries with different complexity levels
test_queries = [
    "¿Que es un CDV?",
    "certificado de vigencia",
    "documento de identidad",
    "formulario de solicitud",
    "requisitos para CDV"
]

for query in test_queries:
    print(f"\n--- Testing Query: '{query}' ---")
    
    try:
        # Perform optimized search
        similar_images = perform_optimized_similarity_search(
            query=query,
            num_results=5,
            similarity_threshold=0.3
        )
        
        print(f"Found {len(similar_images)} relevant images:")
        
        for i, result in enumerate(similar_images):
            print(f"  {i+1}. PDF: {result['pdf_path']}, "
                  f"Page: {result['page_number']}, "
                  f"Score: {result['similarity_score']:.4f}")
        
        if similar_images:
            print(f"✓ Query '{query}' returned relevant results")
        else:
            print(f"⚠ Query '{query}' returned no results above threshold")
            
    except Exception as e:
        print(f"✗ Error testing query '{query}': {e}")
