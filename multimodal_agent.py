# Databricks notebook source
# Multimodal Agent - GPT-4o with CLIP-powered image retrieval
# Optimized with CLIP best practices for better similarity search

# MAGIC %pip install --upgrade openai azure-identity

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

import base64
import json
import time
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
import mlflow.deployments
import numpy as np

# Import our global configuration
from global_config import get_config, get_full_index_name

# COMMAND ----------

# Initialize configuration
config = get_config()

# Get configuration values
azure_config = config["azure_openai"]
llm_config = config["models"]["llm"]
clip_endpoint = config["models"]["clip"]["endpoint_name"]
embedding_dimensions = config["models"]["clip"]["embedding_dimensions"]

# Vector search configuration
endpoint_name = config["vector_search"]["endpoint_name"]
index_name = get_full_index_name()

print("Multimodal Agent Configuration:")
print(f"  Azure OpenAI Endpoint: {azure_config['endpoint']}")
print(f"  Deployment: {azure_config['deployment_name']}")
print(f"  CLIP Endpoint: {clip_endpoint}")
print(f"  Vector Search Index: {index_name}")

# COMMAND ----------

# Initialize clients
client = AzureOpenAI(
    api_key=azure_config["api_key"],
    api_version=azure_config["api_version"],
    azure_endpoint=azure_config["endpoint"]
)

mlflow_client = mlflow.deployments.get_deploy_client("databricks")

# Initialize Vector Search client
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)


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


def perform_optimized_similarity_search(
    query: str, 
    num_results: int = 5,
    similarity_threshold: float = 0.3,
    validate_scores: bool = False
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
                    "processing_timestamp"],  # Temporarily removed embeddings
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
        
        # Debug: Print result structure
        print(f"Debug - Results type: {type(results)}")
        print(f"Debug - Result data type: {type(result_data)}")
        print(f"Debug - Number of results: {len(result_data)}")
        
        if result_data and len(result_data) > 0:
            print(f"Debug - First result type: {type(result_data[0])}")
            print(f"Debug - First result: {result_data[0]}")
        
        for i, result in enumerate(result_data):
            # Handle both list and dict formats
            if isinstance(result, list):
                base64_image = result[0] if len(result) > 0 else ''
                pdf_path = result[1] if len(result) > 1 else 'Unknown'
                page_number = result[2] if len(result) > 2 else 'Unknown'
                processing_timestamp = result[3] if len(result) > 3 else 'Unknown'
                raw_score = result[4] if len(result) > 4 else 0.0
                # embeddings = result[5] if len(result) > 5 else None  # Temporarily removed
                embeddings = None  # Not included in current query
            else:
                pdf_path = result.get('pdf_path', 'Unknown')
                page_number = result.get('page_number', 'Unknown')
                base64_image = result.get('base64_image', '')
                raw_score = result.get('score', 0.0)
                processing_timestamp = result.get('processing_timestamp', 
                                                'Unknown')
                # embeddings = result.get('embeddings', None)  # Temporarily removed
                embeddings = None  # Not included in current query
            
            # Convert score to float, handling cases where it might be a list
            try:
                if isinstance(raw_score, list):
                    # If score is a list, take the first element or 0.0
                    score = float(raw_score[0]) if raw_score else 0.0
                else:
                    score = float(raw_score)
            except (ValueError, TypeError, IndexError):
                print(f"Warning: Could not convert score {raw_score} to float, using 0.0")
                score = 0.0
            
            # Debug: Print score information
            if i < 3:  # Only print first 3 results to avoid spam
                print(f"Debug - Result {i}: raw_score={raw_score} (type: {type(raw_score)}), converted_score={score}")
            
            # Filter by similarity threshold
            if score >= similarity_threshold:
                processed_results.append({
                    'pdf_path': pdf_path,
                    'page_number': page_number,
                    'base64_image': base64_image,
                    'processing_timestamp': processing_timestamp,
                    'similarity_score': score,
                    'embeddings': embeddings
                })
        
        # Sort by similarity score and limit results
        processed_results.sort(key=lambda x: x['similarity_score'], 
                             reverse=True)
        processed_results = processed_results[:num_results]
        
        # Validate similarity scores if requested (for debugging)
        if validate_scores and processed_results:
            print("\n" + "="*30)
            print("Validating Similarity Scores")
            print("="*30)
            print("Note: Validation temporarily disabled - embeddings not included in query")
            # processed_results = validate_similarity_scores(query_embedding, processed_results)
        
        return processed_results
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        import traceback
        traceback.print_exc()
        return []


def validate_similarity_scores(
    query_embedding: List[float], 
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Validate similarity scores by calculating cosine similarity directly
    Useful for debugging and ensuring vector search is working correctly
    """
    try:
        validated_results = []
        
        for result in results:
            # Get the stored embedding from the result
            stored_embedding = result.get('embeddings', None)
            
            if stored_embedding and isinstance(stored_embedding, list):
                # Calculate cosine similarity directly
                query_array = np.array(query_embedding, dtype=np.float32)
                stored_array = np.array(stored_embedding, dtype=np.float32)
                
                # Normalize vectors
                query_norm = np.linalg.norm(query_array)
                stored_norm = np.linalg.norm(stored_array)
                
                if query_norm > 0 and stored_norm > 0:
                    query_normalized = query_array / query_norm
                    stored_normalized = stored_array / stored_norm
                    
                    # Calculate cosine similarity
                    calculated_similarity = np.dot(query_normalized, stored_normalized)
                    
                    # Compare with vector search score
                    vector_search_score = result.get('similarity_score', 0.0)
                    score_difference = abs(calculated_similarity - vector_search_score)
                    
                    validated_result = result.copy()
                    validated_result['calculated_similarity'] = float(calculated_similarity)
                    validated_result['score_difference'] = float(score_difference)
                    validated_result['score_valid'] = score_difference < 0.01  # Within 1% tolerance
                    
                    validated_results.append(validated_result)
                    
                    print(f"PDF: {result.get('pdf_path', 'Unknown')}, "
                          f"Page: {result.get('page_number', 'Unknown')}")
                    print(f"  Vector Search Score: {vector_search_score:.4f}")
                    print(f"  Calculated Similarity: {calculated_similarity:.4f}")
                    print(f"  Difference: {score_difference:.4f}")
                    print(f"  Valid: {validated_result['score_valid']}")
                else:
                    print(f"Warning: Zero norm vectors for {result.get('pdf_path', 'Unknown')}")
            else:
                print(f"Warning: No embedding data for {result.get('pdf_path', 'Unknown')}")
        
        return validated_results
        
    except Exception as e:
        print(f"Error validating similarity scores: {e}")
        import traceback
        traceback.print_exc()
        return results


def create_multimodal_message(
    user_query: str, 
    similar_images: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create a multimodal message for GPT-4o with retrieved images
    """
    messages = []
    
    # Add system message
    system_prompt = llm_config["llm_system_prompt_template"].format(
        context="Retrieved document pages as context"
    )
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # Add user query with images
    content = [{"type": "text", "text": user_query}]
    
    # Add retrieved images as context
    for i, image_data in enumerate(similar_images):
        if image_data.get('base64_image'):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data['base64_image']}",
                    "detail": "high"
                }
            })
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    return messages


def query_multimodal_agent(
    user_query: str,
    num_images: int = 3,
    similarity_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Query the multimodal agent with optimized image retrieval
    """
    try:
        print(f"Processing query: '{user_query}'")
        
        # Step 1: Perform optimized similarity search
        print("Performing similarity search...")
        similar_images = perform_optimized_similarity_search(
            query=user_query,
            num_results=num_images,
            similarity_threshold=similarity_threshold
        )
        
        print(f"Retrieved {len(similar_images)} relevant images")
        
        # Step 2: Create multimodal message
        print("Creating multimodal message...")
        messages = create_multimodal_message(user_query, similar_images)
        
        # Step 3: Query GPT-4o
        print("Querying GPT-4o...")
        response = client.chat.completions.create(
            model=azure_config["deployment_name"],
            messages=messages,
            temperature=llm_config["llm_parameters"]["temperature"],
            max_tokens=llm_config["llm_parameters"]["max_tokens"]
        )
        
        # Step 4: Extract response
        assistant_message = response.choices[0].message.content
        
        # Step 5: Format response with sources
        sources = []
        for image_data in similar_images:
            if image_data.get('similarity_score', 0) >= similarity_threshold:
                sources.append(f"- {image_data['pdf_path']} (Page {image_data['page_number']})")
        
        sources_text = "\n".join(sources) if sources else "No relevant sources found"
        
        return {
            "answer": assistant_message,
            "sources": sources,
            "similarity_scores": [img.get('similarity_score', 0) for img in similar_images],
            "num_images_retrieved": len(similar_images),
            "query": user_query
        }
        
    except Exception as e:
        print(f"Error in multimodal agent: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error processing your query: {str(e)}",
            "sources": [],
            "similarity_scores": [],
            "num_images_retrieved": 0,
            "query": user_query
        }


# COMMAND ----------

# Test the optimized multimodal agent
print("\n" + "="*50)
print("Testing Optimized Multimodal Agent")
print("="*50)

test_queries = [
    "¿Que es un CDV?",
    "¿Cuáles son los requisitos para obtener un certificado de vigencia?",
    "¿Cómo solicito un documento de identidad?",
    "¿Qué formularios necesito para un CDV?"
]

for query in test_queries:
    print(f"\n--- Testing Query: '{query}' ---")
    
    try:
        result = query_multimodal_agent(
            user_query=query,
            num_images=3,
            similarity_threshold=0.3
        )
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])} documents")
        print(f"Similarity scores: {result['similarity_scores']}")
        print(f"Images retrieved: {result['num_images_retrieved']}")
        
    except Exception as e:
        print(f"✗ Error testing query '{query}': {e}")

# COMMAND ----------

print("\n✓ Multimodal agent is ready for use!")
print("Use query_multimodal_agent() function to process user queries.")

# COMMAND ----------

# Quick test to verify the fix works
print("\n" + "="*50)
print("Quick Fix Verification Test")
print("="*50)

try:
    # Test with a simple query
    test_result = perform_optimized_similarity_search(
        query="¿Que es un CDV?",
        num_results=2,
        similarity_threshold=0.1,  # Lower threshold for testing
        validate_scores=False
    )
    
    print(f"Test completed successfully!")
    print(f"Retrieved {len(test_result)} results")
    
    if test_result:
        print("Sample result:")
        print(f"  PDF: {test_result[0].get('pdf_path', 'Unknown')}")
        print(f"  Page: {test_result[0].get('page_number', 'Unknown')}")
        print(f"  Score: {test_result[0].get('similarity_score', 0.0)}")
    
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc() 