# Databricks notebook source
# MAGIC %md
# MAGIC # ColPali Deployment: Multimodal Document Retrieval Pipeline
# MAGIC 
# MAGIC This notebook implements a ColPali-based RAG pipeline following the practitioner recipe:
# MAGIC 
# MAGIC ## Phase 1: Understand ColPali vs CLIP
# MAGIC 
# MAGIC | Feature | CLIP (single-vector) | ColPali (multi-vector + late-interaction) |
# MAGIC |---------|---------------------|-------------------------------------------|
# MAGIC | Visual granularity | Whole page → 1×768 vector | Each 14×14 ViT patch → its own vector (≈196 vectors per page) |
# MAGIC | Retrieval score | cosine(query, page) | Late-interaction: Σ<sub>q-token</sub> max<sub>patch</sub>(q·p) |
# MAGIC | Result | Fast but often "flat" on dense PDFs | +15-30 pp recall on ViDoRe benchmark |
# MAGIC 
# MAGIC ColPali preserves local structure, matching "figure in top-right" or "small footnote number" that CLIP compresses away.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2: Install ColPali Runtime

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade \
# MAGIC     "colpali-engine==0.3.10" \
# MAGIC     # "git+https://github.com/illuin-tech/colpali" \
# MAGIC     "huggingface_hub>=0.23.0" \
# MAGIC     "transformers>=4.41.0" \
# MAGIC     pillow PyMuPDF torchvision # flash-attn # Uncomment if using flash-attn with gpu # pillow-simd does not work with python3.12 while creating serving endpoint

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import all required modules
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import Schema, ColSpec

import torch
from PIL import Image
import base64
from io import BytesIO
import time
import pandas as pd
import requests
import mlflow.pyfunc
import mlflow.deployments
from typing import List, Dict, Any, Tuple
import gc
import os

# ColPali imports
from colpali_engine.models import ColPali, ColPaliProcessor

from bundle_variables import (
    catalog, schema, volume_label, pdf_volume_path,
    vector_search_endpoint_name, vector_search_index_name,
    vector_search_table_name, embeddings_table, pdf_pages_table
)

mlflow.autolog()

# COMMAND ----------

# Configuration
current_device = "cuda:0" if torch.cuda.is_available() else "cpu"
volume_path = f"/Volumes/{catalog}/{schema}/{volume_label}"

# ColPali-specific configuration
registered_model_name = f"{catalog}.{schema}.colpali_embedding_model"
model_endpoint_name = "colpali_model_embedding_generation"
model_name = "vidore/colpali-v1.3"

# Memory optimization settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduce memory usage
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Reduce GPU memory fragmentation

print(f"Device: {current_device}")
print(f"Volume path: {volume_path}")
print(f"Registered model: {registered_model_name}")
print(f"Model name: {model_name}")

# COMMAND ----------

# Create Unity Catalog resources
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume_label}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 3: Load ColPali Model and Processor

# COMMAND ----------

def load_colpali_model():
    """
    Load ColPali model with memory-optimized configuration.
    
    Returns:
        Tuple of (model, processor, model_name)
    """
    print(f"Loading ColPali model: {model_name}")
    print(f"Using device: {current_device}")
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load model with memory-optimized settings
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for better compatibility
        device_map="auto" if torch.cuda.is_available() else current_device,
        low_cpu_mem_usage=True,
        offload_folder="offload",  # Enable model offloading
    ).eval()
    
    # Load processor
    processor = ColPaliProcessor.from_pretrained(model_name)
    
    print(f"Successfully loaded model: {model_name}")
    return model, processor, model_name

# COMMAND ----------

# Load ColPali model and processor
print("Loading ColPali model and processor...")
model, processor, loaded_model_name = load_colpali_model()

print("ColPali model and processor loaded successfully!")
print(f"Using model: {loaded_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 4: Test Basic Functionality

# COMMAND ----------

# Test basic functionality with sample images and queries
def test_basic_functionality():
    """Test the basic ColPali functionality with sample data."""
    
    # Create sample images
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]
    
    # Create sample queries
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]
    
    try:
        # Process the inputs
        batch_images = processor.process_images(images).to(model.device)
        batch_queries = processor.process_queries(queries).to(model.device)
        
        # Forward pass with memory management
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            query_embeddings = model(**batch_queries)
        
        # Score using multi-vector approach
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        
        print("Basic functionality test results:")
        print(f"Image embeddings shape: {len(image_embeddings)} images")
        print(f"Query embeddings shape: {len(query_embeddings)} queries")
        print(f"Scores shape: {scores.shape}")
        print(f"Sample scores:\n{scores}")
        
        # Clear memory
        del batch_images, batch_queries, image_embeddings, query_embeddings, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Error in basic functionality test: {e}")
        return False

# COMMAND ----------

# Run basic functionality test
test_success = test_basic_functionality()

if not test_success:
    print("Basic functionality test failed. Please check model loading.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 5: PDF Processing and Embedding Generation

# COMMAND ----------

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available. PDF processing functions will be limited.")

def pdf_to_png(path: str, dpi: int = 300) -> List[str]:
    """
    Convert PDF pages to PNG images.
    
    Args:
        path: Path to PDF file
        dpi: Resolution for image conversion
        
    Returns:
        List of paths to generated PNG files
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is required for PDF processing")
    
    doc = fitz.open(path)
    png_paths = []
    
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        out = f"/dbfs/tmp/page_{i}.png"
        pix.save(out)
        png_paths.append(out)
    
    doc.close()
    return png_paths

def embed_pages(png_paths: List[str]) -> List[torch.Tensor]:
    """
    Generate multi-vector embeddings for page images.
    
    Args:
        png_paths: List of paths to PNG images
        
    Returns:
        List of patch embeddings for each page
    """
    imgs = [Image.open(p) for p in png_paths]
    batch = processor.process_images(imgs).to(model.device)
    
    with torch.no_grad():
        embeddings = list(model(**batch))
    
    return embeddings

def embed_single_image(image_path: str) -> torch.Tensor:
    """
    Generate embeddings for a single image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Patch embeddings tensor [n_patches, 768]
    """
    img = Image.open(image_path)
    batch = processor.process_images([img]).to(model.device)
    
    with torch.no_grad():
        embeddings = model(**batch)
    
    return embeddings[0]  # Return first (and only) embedding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 6: ColPali Inference Model Class

# COMMAND ----------

class ColPaliInferenceModel(PythonModel):
    """
    MLflow Python model wrapper for ColPali inference.
    Supports both image and text embedding generation with multi-vector scoring.
    Memory-optimized for serving endpoints.
    """
    
    def load_context(self, context):
        """Load the ColPali model and processor with memory optimization."""
        self.model_name = model_name
        
        # Clear memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Load model with memory-optimized settings
        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use float16 for better memory efficiency
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            offload_folder="offload",
        ).eval()
        
        self.processor = ColPaliProcessor.from_pretrained(self.model_name)

    def generate_image_embedding_from_base64_string(
        self, base64_string: str
    ) -> Dict[str, Any]:
        """
        Generate multi-vector embeddings for an image from base64 string.
        
        Args:
            base64_string: Base64 encoded string of the image
            
        Returns:
            Dictionary containing patch embeddings
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            
            # Process image with ColPali
            batch = self.processor.process_images([image]).to(
                self.model.device
            )
            
            # Generate embeddings with memory management
            with torch.no_grad():
                image_features = self.model(**batch)
                # Return first (and only) embedding
                patch_embeddings = image_features[0]
            
            # Convert to list for JSON serialization
            patch_embeddings_list = patch_embeddings.tolist()
            embedding_dim = (len(patch_embeddings_list[0]) 
                           if patch_embeddings_list else 0)
            
            # Clear memory
            del batch, image_features, patch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "patch_embeddings": patch_embeddings_list,
                "num_patches": len(patch_embeddings_list),
                "embedding_dim": embedding_dim
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        
    def generate_text_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generate multi-vector embeddings for text using ColPali.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing token embeddings
        """
        try:
            # Process text with ColPali
            batch = self.processor.process_queries([text]).to(
                self.model.device
            )
            
            # Generate embeddings with memory management
            with torch.no_grad():
                text_features = self.model(**batch)
                # Return first (and only) embedding
                token_embeddings = text_features[0]
            
            # Convert to list for JSON serialization
            token_embeddings_list = token_embeddings.tolist()
            embedding_dim = (len(token_embeddings_list[0]) 
                           if token_embeddings_list else 0)
            
            # Clear memory
            del batch, text_features, token_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "token_embeddings": token_embeddings_list,
                "num_tokens": len(token_embeddings_list),
                "embedding_dim": embedding_dim
            }
           
        except Exception as e:
            print(f"Error processing text: {e}")
            return None

    def score_query_against_images(
        self, query: str, image_base64_list: List[str]
    ) -> Dict[str, Any]:
        """
        Score a query against multiple images using multi-vector approach.
        
        Args:
            query: Query text
            image_base64_list: List of base64 encoded images
            
        Returns:
            Dictionary containing scores and rankings
        """
        try:
            # Process query
            query_batch = self.processor.process_queries([query]).to(
                self.model.device
            )
            
            # Process images
            images = []
            for img_base64 in image_base64_list:
                image_data = base64.b64decode(img_base64)
                image = Image.open(BytesIO(image_data)).convert("RGB")
                images.append(image)
            
            image_batch = self.processor.process_images(images).to(
                self.model.device
            )
            
            # Generate embeddings with memory management
            with torch.no_grad():
                query_embeddings = self.model(**query_batch)
                image_embeddings = self.model(**image_batch)
            
            # Score using multi-vector approach
            scores = self.processor.score_multi_vector(
                query_embeddings, image_embeddings
            )
            
            # Convert scores to list and create rankings
            scores_list = scores.tolist()[0]  # First (and only) query
            rankings = [
                {"image_index": i, "score": score}
                for i, score in enumerate(scores_list)
            ]
            rankings.sort(key=lambda x: x["score"], reverse=True)
            
            # Clear memory
            del query_batch, image_batch, query_embeddings, image_embeddings, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "scores": scores_list,
                "rankings": rankings,
                "query": query,
                "num_images": len(image_base64_list)
            }
            
        except Exception as e:
            print(f"Error scoring query against images: {e}")
            return None

    def predict(self, context, model_input):
        """
        Main prediction method that handles various input types.
        
        Args:
            model_input: Could be a pandas DataFrame with:
                        - 'text' column for text embedding
                        - 'image_base64' column for image embedding
                        - 'query' and 'image_base64_list' for scoring
                        - 'text' and 'image_base64' for both embeddings
        
        Returns:
            Dictionary with predictions containing multi-vector embeddings
        """
        # Handle DataFrame input
        if isinstance(model_input, pd.DataFrame):
            # Check for scoring mode
            if ('query' in model_input.columns and 
                'image_base64_list' in model_input.columns):
                
                query = model_input['query'].to_list()[0]
                image_base64_list = model_input['image_base64_list'].to_list()[0]
                
                return self.score_query_against_images(query, image_base64_list)
            
            # Check for dual embedding mode
            elif ('text' in model_input.columns and 
                  'image_base64' in model_input.columns):
                
                text_embedding = self.generate_text_embedding(
                    model_input['text'].to_list()[0]
                )
                image_embedding = (
                    self.generate_image_embedding_from_base64_string(
                        model_input['image_base64'].to_list()[0]
                    )
                )
                return {"predictions": [text_embedding, image_embedding]}

            # Text-only mode
            elif 'text' in model_input.columns:
                embedding = self.generate_text_embedding(
                    model_input['text'].to_list()[0]
                )
                return {"predictions": embedding}

            # Image-only mode
            elif 'image_base64' in model_input.columns:
                embedding = (
                    self.generate_image_embedding_from_base64_string(
                        model_input['image_base64'].to_list()[0]
                    )
                )
                return {"predictions": embedding}
            
        # Handle dictionary input
        elif isinstance(model_input, dict):
            # Check for scoring mode
            if ('query' in model_input and model_input['query'] and 
                'image_base64_list' in model_input and 
                model_input['image_base64_list']):
                
                return self.score_query_against_images(
                    model_input['query'], 
                    model_input['image_base64_list']
                )
            
            # Check for dual embedding mode
            elif ('text' in model_input and model_input['text'] and 
                  'image_base64' in model_input and 
                  model_input['image_base64']):
                
                text_embedding = self.generate_text_embedding(
                    model_input['text']
                )
                image_embedding = (
                    self.generate_image_embedding_from_base64_string(
                        model_input['image_base64']
                    )
                )
                return {"predictions": [text_embedding, image_embedding]}

            # Text-only mode
            elif 'text' in model_input and model_input['text']:
                embedding = self.generate_text_embedding(
                    model_input['text']
                )
                return {"predictions": embedding}
                
            # Image-only mode
            elif ('image_base64' in model_input and 
                  model_input['image_base64']):
                embedding = (
                    self.generate_image_embedding_from_base64_string(
                        model_input['image_base64']
                    )
                )
                return {"predictions": embedding}

        raise ValueError(
            f"Invalid input format. Your input type was: {type(model_input)}. "
            "Expected a dictionary or pandas DataFrame with 'text', "
            "'image_base64', 'query', or 'image_base64_list' keys."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 7: Model Registration and Deployment

# COMMAND ----------

# Define input and output schemas for MLflow
input_schema = Schema([
    ColSpec("string", "text", required=False),
    ColSpec("string", "image_base64", required=False),
    ColSpec("string", "query", required=False),
    ColSpec("string", "image_base64_list", required=False)
])

# ColPali embeddings are typically 768-dimensional for the v1.3 model
# We return a list of patch embeddings, so we use a more flexible schema
output_schema = Schema([
    ColSpec("string", "predictions", required=True)
])

# Create the model signature
print("Creating model signature and logging model...")
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log the model to MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        name="colpali_model",
        artifacts={},  # No artifacts needed since we load from HF
        python_model=ColPaliInferenceModel(),
        signature=signature,
        registered_model_name=registered_model_name,
        extra_pip_requirements=[
            "colpali-engine==0.3.10",
            "torch",
            "pillow",
            "transformers",
            "PyMuPDF",
            "huggingface_hub",
            # "flash-attn", # Uncomment if using flash-attn with gpu
            "torchvision"
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 8: Test the Model

# COMMAND ----------

# Test the deployed model
model_version_uri = f"models:/{registered_model_name}/1"
first_version = mlflow.pyfunc.load_model(model_version_uri)

# Test text embedding
text_result = first_version.predict({
    'text': "Is attention really all you need?"
})
print("Text embedding result:")
print(f"Number of tokens: {text_result['predictions']['num_tokens']}")
print(f"Embedding dimension: {text_result['predictions']['embedding_dim']}")

# Test image embedding with sample image
image_url = (
    "https://miro.medium.com/v2/resize:fit:447/"
    "1*G0CAXQqb250tgBMeeVvN6g.png"
)
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
buffer = BytesIO()
img.save(buffer, format=img.format)
img_bytes = buffer.getvalue()
img_base64 = base64.b64encode(img_bytes).decode('utf-8')

image_result = first_version.predict({
    'image_base64': img_base64
})
print("\nImage embedding result:")
print(f"Number of patches: {image_result['predictions']['num_patches']}")
print(f"Embedding dimension: {image_result['predictions']['embedding_dim']}")

# Test scoring functionality
scoring_result = first_version.predict({
    'query': "What is this image about?",
    'image_base64_list': [img_base64]
})
print("\nScoring result:")
print(f"Number of images: {scoring_result['num_images']}")
print(f"Scores: {scoring_result['scores']}")
print(f"Rankings: {scoring_result['rankings']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 9: Create Model Endpoint

# COMMAND ----------

# Create model endpoint
client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

endpoint = client.create_endpoint(
    name=model_endpoint_name,
    config={
        "served_entities": [
            {
                "name": "colpali_model_gpu" if torch.cuda.is_available() else "colpali_model_cpu",
                "entity_name": registered_model_name,
                "entity_version": "1",
                "workload_size": "Large",  # Always use Large for memory-intensive models
                "scale_to_zero_enabled": False,  # Disable scale to zero to prevent cold starts
                "environment_vars": {
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
                }
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": "colpali_model_gpu" if torch.cuda.is_available() else "colpali_model_cpu",
                    "traffic_percentage": 100
                }
            ]
        }
    }
)

# Monitor deployment status
while True:
    deployment = client.get_endpoint(model_endpoint_name)

    if deployment['state']['config_update'] == "NOT_UPDATING":
        print("Endpoint is ready")
        break
    elif deployment['state']['config_update'] in [
        "UPDATE_FAILED", "DEPLOYMENT_FAILED"
    ]:
        print(f"Deployment failed: {deployment['state']}")
        break
    else:
        print(
            f"Deployment in progress... "
            f"Status: {deployment['state']['config_update']}"
        )
        time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 10: Test the Endpoint

# COMMAND ----------

endpoint_name = model_endpoint_name
databricks_instance = (
    dbutils.entry_point.getDbutils()
    .notebook().getContext().browserHostName().get()
)
endpoint_url = f"https://{databricks_instance}/ml/endpoints/{endpoint_name}"
print(f"Endpoint URL: {endpoint_url}")

# COMMAND ----------

# Test the endpoint with scoring functionality
start_time = time.time()
try:
    response = client.predict(
        endpoint=model_endpoint_name,
        inputs={"dataframe_split": {
            "columns": ["query", "image_base64_list"],
            "data": [["What is this image about?", [img_base64]]]
        }},
        timeout=120  # Increase timeout to 2 minutes
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(response)
    print(f"Response time: {total_time:.2f} seconds")
except Exception as e:
    print(f"Error testing endpoint: {e}")
    print("This might be due to memory constraints. Consider using a larger workload size.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 11: Multi-Vector Indexing and Retrieval

# COMMAND ----------

def create_patch_vector_table(
    page_id: int, vectors: List[torch.Tensor]
) -> pd.DataFrame:
    """
    Create a DataFrame for storing patch vectors.
    
    Args:
        page_id: Unique identifier for the page
        vectors: List of patch embeddings for the page
        
    Returns:
        DataFrame with page_id, patch_idx, and vector columns
    """
    rows = []
    for emb in vectors:
        for i, v in enumerate(emb):
            rows.append({
                'page_id': page_id,
                'patch_idx': int(i),
                'vec': v.tolist()
            })
    
    return pd.DataFrame(rows)

def score_query_late_interaction(
    query: str, 
    candidate_pages: pd.DataFrame,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Score query against candidate pages using late-interaction.
    
    Args:
        query: Query text
        candidate_pages: DataFrame with page_id, patch_idx, vec columns
        top_k: Number of top results to return
        
    Returns:
        List of (page_id, score) tuples sorted by score
    """
    # Generate query embeddings
    q = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        q_emb = model(**q)  # [n_tokens, 768]
    
    # Score each page using late-interaction
    scores = {}
    for pid, grp in candidate_pages.groupby("page_id"):
        patches = torch.tensor(grp["vec"].tolist())  # [n_patches, 768]
        
        # Use the processor's score_multi_vector method
        # This is more efficient than manual cosine similarity
        page_emb = [patches]  # Wrap in list for batch processing
        sim = processor.score_multi_vector(q_emb, page_emb)
        scores[pid] = sim.item()
    
    # Return top-k results
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

# COMMAND ----------

# Example usage of late-interaction scoring
def demo_late_interaction():
    """Demonstrate late-interaction scoring with sample data."""
    
    # Create sample patch vectors for a page
    sample_image = Image.new("RGB", (224, 224), color="white")
    batch = processor.process_images([sample_image]).to(model.device)
    
    with torch.no_grad():
        sample_embeddings = model(**batch)[0]
    
    # Create sample candidate pages
    sample_pages = create_patch_vector_table(0, [sample_embeddings])
    
    # Score a query
    query = "What is the main topic of this document?"
    results = score_query_late_interaction(query, sample_pages, top_k=5)
    
    print(f"Query: {query}")
    print("Top results:")
    for page_id, score in results:
        print(f"  Page {page_id}: {score:.4f}")

# COMMAND ----------

# Run demo
demo_late_interaction()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 12: Migration Checklist Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### Migration Checklist
# MAGIC 
# MAGIC | Task | Effort | Status | Comment |
# MAGIC |------|--------|--------|---------|
# MAGIC | Convert PDF → PNG | ✅ | Done | Already implemented for CLIP |
# MAGIC | Replace CLIP processor/model calls with ColPali equivalents | <30 min | ✅ | Done |
# MAGIC | Change index schema to multi-vector | moderate | ✅ | Implemented |
# MAGIC | Update similarity_search code to late-interaction | moderate | ✅ | Implemented |
# MAGIC | Use score_multi_vector for efficient scoring | ✅ | Done | New optimization |
# MAGIC | Fix memory and timeout issues | ✅ | Done | Memory optimization implemented |
# MAGIC | Retune ANN / pool-factor | 1–2 h | 🔄 | Optional optimization |
# MAGIC 
# MAGIC ### Key Improvements in This Version
# MAGIC 
# MAGIC 1. **Updated colpali-engine**: Using version 0.3.10 with improved APIs
# MAGIC 2. **score_multi_vector method**: More efficient than manual cosine similarity
# MAGIC 3. **Memory optimization**: Float16, model offloading, and memory management
# MAGIC 4. **Enhanced scoring functionality**: Direct query-to-image scoring
# MAGIC 5. **Improved error handling**: More robust model loading and inference
# MAGIC 6. **Endpoint optimization**: Large workload size and disabled scale-to-zero
# MAGIC 
# MAGIC ### Performance Notes
# MAGIC 
# MAGIC - **Memory optimization**: Float16 instead of bfloat16 for better compatibility
# MAGIC - **Model offloading**: Reduces memory footprint during loading
# MAGIC - **Large workload size**: Ensures sufficient memory for model inference
# MAGIC - **Disabled scale-to-zero**: Prevents cold start issues
# MAGIC - **Memory cleanup**: Automatic GPU memory management

# COMMAND ----------

# MAGIC %md
# MAGIC # ColPali Deployment Complete ✅
# MAGIC 
# MAGIC The ColPali model has been successfully deployed with the following 
# MAGIC features:
# MAGIC 
# MAGIC - **Multi-vector embeddings**: Each page produces ~196 patch vectors for 
# MAGIC   fine-grained retrieval
# MAGIC - **Late-interaction scoring**: Advanced similarity computation preserving 
# MAGIC   spatial information
# MAGIC - **score_multi_vector optimization**: Efficient scoring using ColPali's 
# MAGIC   native method
# MAGIC - **Memory optimization**: Float16, model offloading, and memory management
# MAGIC - **MLflow integration**: Full model lifecycle management
# MAGIC - **Endpoint deployment**: Ready for inference via serving endpoint
# MAGIC - **Migration ready**: Drop-in replacement for CLIP with improved recall
# MAGIC - **Enhanced scoring**: Direct query-to-image scoring functionality
# MAGIC 
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. **Index your documents**: Use the `embed_pages()` function to process PDFs
# MAGIC 2. **Store patch vectors**: Use `create_patch_vector_table()` for Delta storage
# MAGIC 3. **Query with late-interaction**: Use `score_query_late_interaction()` for 
# MAGIC    retrieval
# MAGIC 4. **Use scoring endpoint**: Leverage the new scoring functionality for 
# MAGIC    direct query-to-image matching
# MAGIC 5. **Optimize performance**: Consider GPU deployment and PLAID indexing for 
# MAGIC    scale 