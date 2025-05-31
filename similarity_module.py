# similarity_module.py

import os
import io
from PIL import Image
import numpy as np
import faiss
import pymongo
import torch # You need PyTorch as your original code uses it
import torchvision.transforms as transforms # For your image transformation
import torchvision.models as models # For your model loading

from urllib.parse import quote # For URL encoding product handles

# --- Configuration for MongoDB and Shopify (read from environment variables) ---
# IMPORTANT: For production, these should come from your environment variables (.env file or system env)
MONGO_URI = os.environ.get('MONGO_URI')
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', "zappos_products")
MONGO_COLLECTION_NAME = os.environ.get('MONGO_COLLECTION_NAME', "products")

SHOPIFY_STORE_NAME = os.environ.get('SHOPIFY_STORE_NAME')

# --- Global Variables for Loaded Resources (initialized once) ---
_faiss_index = None
_image_embedding_model = None
_image_transform = None # Your transformation pipeline
_device = None
_faiss_index_to_cid_map = None # This is CRUCIAL: Maps FAISS index to your MongoDB _id (CID)
_mongo_collection = None


def load_faiss_index_and_map(index_path="faiss_index.bin", cid_map_path="cid_map.npy"):
    """
    Loads the pre-trained FAISS index and the corresponding CID mapping.
    This mapping is essential to convert FAISS internal indices to your MongoDB CIDs.
    You MUST ensure 'cid_map.npy' contains the CIDs in the same order as your embeddings
    were added to the FAISS index.
    """
    global _faiss_index, _faiss_index_to_cid_map
    if _faiss_index is None:
        print(f"Loading FAISS index from {index_path}...")
        try:
            _faiss_index = faiss.read_index(index_path)
            print(f"FAISS index loaded. Total vectors: {_faiss_index.ntotal}")

            # Load the CID mapping
            # This file (cid_map.npy) should contain a numpy array of your MongoDB _id's (CIDs)
            # in the exact same order as you added their embeddings to the FAISS index.
            if os.path.exists(cid_map_path):
                _faiss_index_to_cid_map = np.load(cid_map_path, allow_pickle=True)
                print(f"CID mapping loaded. Total CIDs: {len(_faiss_index_to_cid_map)}")
                if len(_faiss_index_to_cid_map) != _faiss_index.ntotal:
                    print("WARNING: CID map length does not match FAISS index size!")
            else:
                raise FileNotFoundError(f"CID map file not found at {cid_map_path}. Cannot map FAISS indices to CIDs.")

        except Exception as e:
            print(f"Error loading FAISS index or CID map: {e}")
            _faiss_index = None
            _faiss_index_to_cid_map = None
    return _faiss_index, _faiss_index_to_cid_map

def load_image_embedding_model_and_transform():
    """
    Loads your pre-trained image embedding model and defines the transformation pipeline.
    This replaces your `model`, `transform`, and `device` variables from Colab.
    """
    global _image_embedding_model, _image_transform, _device
    if _image_embedding_model is None:
        print("Loading image embedding model and transforms...")
        try:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {_device}")

            # Replace with your actual model architecture (e.g., models.resnet50)
            # Ensure 'model' is your actual model variable
            _image_embedding_model = models.resnet50(pretrained=True) # Example: ResNet50
            # Remove the last classification layer if it's not part of your feature extractor
            # For ResNet, typically you want features before the final fc layer
            _image_embedding_model = torch.nn.Sequential(*list(_image_embedding_model.children())[:-1])
            _image_embedding_model.eval() # Set to evaluation mode
            _image_embedding_model.to(_device) # Move model to device

            _image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("Image embedding model and transforms loaded successfully.")
        except Exception as e:
            print(f"Error loading image embedding model/transforms: {e}")
            _image_embedding_model = None
            _image_transform = None
            _device = None
    return _image_embedding_model, _image_transform, _device

def get_mongo_collection():
    """
    Establishes and returns the MongoDB collection object.
    """
    global _mongo_collection
    if _mongo_collection is None:
        if not MONGO_URI:
            raise ValueError("MONGO_URI environment variable not set.")
        print("Connecting to MongoDB...")
        try:
            client = pymongo.MongoClient(MONGO_URI)
            db = client[MONGO_DB_NAME]
            _mongo_collection = db[MONGO_COLLECTION_NAME]
            print("Connected to MongoDB successfully.")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            _mongo_collection = None
    return _mongo_collection

def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Processes the PIL Image using the loaded transform and model,
    and returns its embedding vector.
    This replaces the 'Load and transform query image' and 'Extract feature' parts.
    """
    if _image_embedding_model is None or _image_transform is None or _device is None:
        raise ValueError("Image embedding model or transforms are not loaded.")

    try:
        # Apply transformations
        img_t = _image_transform(image)
        batch_t = torch.unsqueeze(img_t, 0) # Add batch dimension
        batch_t = batch_t.to(_device) # Move tensor to device

        # Extract feature
        with torch.no_grad():
            query_feature = _image_embedding_model(batch_t).cpu().numpy().flatten().astype('float32').reshape(1, -1)
        
        return query_feature

    except Exception as e:
        raise RuntimeError(f"Error generating image embedding: {e}")

def search_faiss(query_embedding: np.ndarray, k: int = 5):
    """
    Performs a similarity search using FAISS and maps indices to CIDs.
    This replaces the 'Search in FAISS index' part.
    """
    if _faiss_index is None or _faiss_index_to_cid_map is None:
        raise ValueError("FAISS index or CID map is not loaded.")
    
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1) # Ensure 2D for FAISS search

    D, I = _faiss_index.search(query_embedding, k) # D: distances, I: indices

    results = []
    # I[0] contains the actual indices from FAISS for the first (and only) query
    for i, dist in zip(I[0], D[0]):
        if i >= 0 and i < len(_faiss_index_to_cid_map): # Ensure valid index
            cid = _faiss_index_to_cid_map[i]
            results.append({"cid": str(cid), "distance": float(dist)})
        else:
            print(f"Warning: FAISS returned invalid index {i}")
            
    # Filter out query image if it's in the results (distance very close to 0)
    # This is a common requirement in similarity search.
    # We allow a small tolerance (1e-6) to account for floating point inaccuracies.
    # Note: If your query image is *not* in your FAISS index, this filter won't remove anything.
    filtered_results = [res for res in results if res['distance'] > 1e-6]
    
    # If filtering removed items, and we need more, we'd need to re-search with k+N.
    # For now, we return what's filtered.
    return filtered_results


def get_product_details_from_mongo(cids: list) -> list:
    """
    Fetches detailed product information from MongoDB for a list of CIDs.
    Constructs Shopify product URLs.
    This replaces the 'Retrieve and display similar images' part.
    """
    if _mongo_collection is None:
        raise ValueError("MongoDB collection is not connected.")
    if not SHOPIFY_STORE_NAME:
        raise ValueError("SHOPIFY_STORE_NAME environment variable not set.")

    product_details = []
    
    # Query MongoDB for documents where _id is in the list of CIDs
    cursor = _mongo_collection.find({"_id": {"$in": cids}})
    
    for doc in cursor:
        product_id_mongo = str(doc.get('_id'))
        cloudinary_url = doc.get('cloudinary_url')
        
        # Reconstruct product title using the same logic as Shopify uploader
        product_title = generate_product_title(
            doc.get('gender'),
            doc.get('material'),
            doc.get('SubCategory')
        )
        if not product_title:
            product_title = f"Zappos Product {product_id_mongo}"

        # Reconstruct product handle using the same logic as Shopify uploader
        # The `quote` function ensures special characters in the handle are correctly encoded for URLs.
        product_handle = sanitize_for_handle(f"{product_id_mongo}-{product_title}")
        shopify_product_url = f"https://{SHOPIFY_STORE_NAME}.myshopify.com/products/{quote(product_handle)}"

        product_details.append({
            "CID": product_id_mongo,
            "title": product_title,
            "cloudinary_url": cloudinary_url,
            "shopify_url": shopify_product_url,
            "category": doc.get('category'),
            "SubCategory": doc.get('SubCategory'),
            # Add any other relevant fields you want to return to the frontend
        })
    return product_details

# Helper functions (copied from previous Shopify script, needed for handle/title generation)
def sanitize_for_handle(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace(" ", "-")
    text = "".join(char for char in text if char.isalnum() or char == '-')
    return text

def generate_product_title(gender, material, subcategory):
    gender_prefix = ""
    if gender and isinstance(gender, str):
        if gender.lower() == "men":
            gender_prefix = "Men's "
        elif gender.lower() == "women":
            gender_prefix = "Women's "
        elif gender.lower() == "unisex":
            gender_prefix = "Unisex "

    material_name = material if material and isinstance(material, str) else ""
    subcategory_name = subcategory if subcategory and isinstance(subcategory, str) else ""

    if material_name:
        material_name = material_name.capitalize()
    if subcategory_name:
        subcategory_name = subcategory_name.capitalize()

    title_parts = [gender_prefix, material_name, subcategory_name]
    return " ".join(filter(None, title_parts)).strip()