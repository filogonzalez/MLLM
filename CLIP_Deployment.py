# Databricks notebook source
# MAGIC %pip install --upgrade transformers torch==2.6.0 requests torchvision
# MAGIC %pip install --upgrade databricks-sdk mlflow pillow numpy scikit-learn
# MAGIC %pip install --upgrade accelerate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import all required modules at the top
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import Schema, ColSpec, TensorSpec

from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import torch
import time
import pandas as pd
import requests
import mlflow.pyfunc
import mlflow.deployments
import traceback

from bundle_variables import (
    catalog, schema, model_name, model_endpoint_name,
    registered_model_name, volume_label
)

mlflow.autolog()

# COMMAND ----------

volume_path = f"/Volumes/{catalog}/{schema}/{volume_label}"
print(f"Volume name: {volume_path}")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume_label}")

# COMMAND ----------

# Use CPU for stable deployment - can be changed to GPU if needed
clip_model_name = "openai/clip-vit-base-patch32"

# Load CLIP model and processor
model = CLIPModel.from_pretrained(
    clip_model_name,
    cache_dir=volume_path
)
model.eval()

processor = CLIPProcessor.from_pretrained(
    clip_model_name,
    cache_dir=volume_path
)

# COMMAND ----------


class CLIPInferenceModel(PythonModel):
    def load_context(self, context):
        """Load the CLIP model and processor from cached artifacts."""
        self.clip_model_name = "openai/clip-vit-base-patch32"

        # Load model from cached artifacts
        self.model = CLIPModel.from_pretrained(
            self.clip_model_name,
            cache_dir=context.artifacts['cache']
        )
        self.model.eval()
        
        self.processor = CLIPProcessor.from_pretrained(
            self.clip_model_name,
            cache_dir=context.artifacts['cache']
        )

    def generate_image_embedding_from_base64_string(self, base64_string):
        """
        Generate embeddings for an image from a base64 encoded string
        using the CLIP model.
        
        Args:
            base64_string: Base64 encoded string of the image
            
        Returns:
            Dictionary containing flattened image embedding
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            
            # Process image with CLIP
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            )

            # Generate image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the features (standard for CLIP)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
            
            # Convert to list for JSON serialization
            image_embedding_flat = image_features.squeeze().tolist()
            return {"embedding": image_embedding_flat}
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        
    def generate_text_embedding(self, text):
        """
        Generate embeddings for text using the CLIP model.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing flattened text embedding
        """
        try:
            # Process text with CLIP
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Generate text embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize the features (standard for CLIP)
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )
            
            # Convert to list for JSON serialization
            text_embedding_flat = text_features.squeeze().tolist()
            return {"embedding": text_embedding_flat}
           
        except Exception as e:
            print(f"Error processing text: {e}")
            return None

    def predict(self, context, model_input):
        """
        Main prediction method that handles both text and image inputs.
        
        Args:
            model_input: Could be a pandas DataFrame with either a 'text'
                        column or an 'image_base64' column (base64 string).
        
        Returns:
            Dictionary with predictions containing embeddings
        """
        # Handle DataFrame input
        if isinstance(model_input, pd.DataFrame):
            if ('text' in model_input.columns and 
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

            elif 'text' in model_input.columns:
                embedding = self.generate_text_embedding(
                    model_input['text'].to_list()[0]
                )
                return {"predictions": embedding}

            elif 'image_base64' in model_input.columns:
                embedding = (
                    self.generate_image_embedding_from_base64_string(
                        model_input['image_base64'].to_list()[0]
                    )
                )
                return {"predictions": embedding}
            
        # Handle dictionary input
        elif isinstance(model_input, dict):
            if ('text' in model_input and model_input['text'] and 
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

            elif 'text' in model_input and model_input['text']:
                embedding = self.generate_text_embedding(
                    model_input['text']
                )
                return {"predictions": embedding}
                
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
            "Expected a dictionary or pandas DataFrame with 'text' or "
            "'image_base64' keys."
        )

# COMMAND ----------

# Define input and output schemas for MLflow
input_schema = Schema([
    ColSpec("string", "text", required=False),
    ColSpec("string", "image_base64", required=False)
])

# CLIP embeddings are typically 512-dimensional for the base model
output_schema = Schema([
    TensorSpec(np.dtype("float32"), (512,), "embedding"),
])

# Create the model signature
print("Creating model signature and logging model...")
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log the model to MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        name="clip_model",
        artifacts={'cache': volume_path},
        python_model=CLIPInferenceModel(),
        signature=signature,
        registered_model_name=registered_model_name,
        extra_pip_requirements=[
            "transformers",
            "torch",
            "torchvision",
            "pillow"
        ]
    )

# COMMAND ----------

# Test the model with a sample image
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

# COMMAND ----------

# Test the deployed model
model_version_uri = f"models:/{registered_model_name}/1"
first_version = mlflow.pyfunc.load_model(model_version_uri)
result = first_version.predict({
    'text': "Is attention really all you need?",
    'image_base64': img_base64
})
print("Model test result:")
print(result)

# COMMAND ----------

# Create model endpoint
client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

endpoint = client.create_endpoint(
    name=model_endpoint_name,
    config={
        "served_entities": [
            {
                "name": model_name,
                "entity_name": registered_model_name,
                "entity_version": "1",
                "workload_size": "Medium",
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": model_name,
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
# MAGIC # Test the endpoint

# COMMAND ----------

endpoint_name = model_endpoint_name
databricks_instance = dbutils.entry_point.getDbutils().notebook().getContext().browserHostName().get()
endpoint_url = f"https://{databricks_instance}/ml/endpoints/{endpoint_name}"
print(f"Endpoint URL: {endpoint_url}")

# COMMAND ----------

image_url = "https://miro.medium.com/v2/resize:fit:447/1*G0CAXQqb250tgBMeeVvN6g.png"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
buffer = BytesIO()
img.save(buffer, format=img.format)
img_bytes = buffer.getvalue()

img_base64 = base64.b64encode(img_bytes).decode('utf-8')

# COMMAND ----------

start_time = time.time()
response = client.predict(
            endpoint=model_endpoint_name,
            inputs={"dataframe_split": {
                    "columns": ["text", "image_base64"],
                    "data": [["this is just a test", img_base64]]
                    }
            }
          )
end_time = time.time()
total_time = end_time-start_time
print(response)
print(f"Final Time: {total_time}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Deployment Complete
# MAGIC
# MAGIC The CLIP model has been successfully deployed with the following features:
# MAGIC - **Multimodal embeddings**: Support for both text and image inputs
# MAGIC - **CPU-optimized**: Configured for stable CPU deployment
# MAGIC - **Volume storage**: Model cached in DBFS volume for efficient loading
# MAGIC - **MLflow integration**: Full model lifecycle management
# MAGIC - **Endpoint deployment**: Ready for inference via serving endpoint

# COMMAND ----------

