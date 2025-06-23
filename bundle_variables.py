"""
Bundle Variables Module
Loads configuration variables from bundle.variables.yml file
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_bundle_variables() -> Dict[str, Any]:
    """Load bundle variables from bundle.variables.yml file"""
    try:
        # Try to find bundle.variables.yml in current or parent directories
        current_dir = Path.cwd()
        bundle_vars_file = None
        
        # Look for bundle.variables.yml in current and parent directories
        for path in [current_dir] + list(current_dir.parents):
            potential_file = path / "bundle.variables.yml"
            if potential_file.exists():
                bundle_vars_file = potential_file
                break
        
        if bundle_vars_file is None:
            print("Warning: bundle.variables.yml not found. "
                  "Using empty variables dict.")
            return {}
        
        with open(bundle_vars_file, 'r') as file:
            bundle_config = yaml.safe_load(file)
            variables = bundle_config.get('variables', {})
            
            # Extract default values from the nested structure
            extracted_vars = {}
            for var_name, var_config in variables.items():
                if isinstance(var_config, dict) and 'default' in var_config:
                    extracted_vars[var_name] = var_config['default']
                else:
                    extracted_vars[var_name] = var_config
            
            return extracted_vars
    except Exception as e:
        print(f"Warning: Could not load bundle variables: {e}")
        return {}


# Load all bundle variables
_variables = load_bundle_variables()


# Export all variables as module-level attributes
for var_name, var_value in _variables.items():
    globals()[var_name] = var_value


# Also provide a function to get all variables as a dict
def get_all_variables() -> Dict[str, Any]:
    """Get all bundle variables as a dictionary"""
    return _variables.copy()


# Export specific commonly used variables for convenience
if 'catalog' in _variables:
    catalog = _variables['catalog']
if 'schema' in _variables:
    schema = _variables['schema']
# if 'volume_path' in _variables:
#     volume_path = _variables['volume_path']
if 'volume_label' in _variables:
    volume_label = _variables['volume_label']
if 'pdf_volume_path' in _variables:
    pdf_volume_path = _variables['pdf_volume_path']
if 'model_name' in _variables:
    model_name = _variables['model_name']
if 'endpoint_name' in _variables:
    endpoint_name = _variables['endpoint_name']
if 'model_registered_name' in _variables:
    model_registered_name = _variables['model_registered_name']
if 'llm_endpoint_name' in _variables:
    llm_endpoint_name = _variables['llm_endpoint_name']
if 'azure_endpoint' in _variables:
    azure_endpoint = _variables['azure_endpoint']
if 'azure_deployment_name' in _variables:
    azure_deployment_name = _variables['azure_deployment_name']
if 'vector_search_endpoint_name' in _variables:
    vector_search_endpoint_name = _variables['vector_search_endpoint_name']
if 'vector_search_index_name' in _variables:
    vector_search_index_name = _variables['vector_search_index_name']
if 'vector_search_table_name' in _variables:
    vector_search_table_name = _variables['vector_search_table_name']
if 'pdf_pages_table' in _variables:
    pdf_pages_table = _variables['pdf_pages_table']
if 'embeddings_table' in _variables:
    embeddings_table = _variables['embeddings_table']
if 'vector_index_table' in _variables:
    vector_index_table = _variables['vector_index_table']

# Convenience variables for CLIP deployment
if 'model_name' in _variables:
    model_name = "model_cpu"
if 'endpoint_name' in _variables:
    model_endpoint_name = _variables['endpoint_name']
if 'model_registered_name' in _variables:
    registered_model_name = _variables['model_registered_name'] 