# create_faiss_index.py

import os
import pymongo
import requests
from PIL import Image
import io
import numpy as np
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MONGO_URI = os.environ.get('MONGO_URI')
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', "zappos_products")
MONGO_COLLECTION_NAME = os.environ.get('MONGO_COLLECTION_NAME', "products")

FAISS_INDEX_FILE = "faiss_index.bin"
CID_MAP_FILE = "cid_map.npy"

# --- Device Configuration (CPU or GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for embeddings: {device}")

# --- 1. Load Pre-trained Image Embedding Model and Transforms ---
print("\n--- 1. Loading Image Embedding Model ---")
try:
    # Load a pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    # Remove the final classification layer to get feature embeddings
    # For ResNet, features are typically before the final fully connected layer (fc)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval() # Set model to evaluation mode
    model.to(device) # Move model to the selected device

    # Define the image transformations required by the model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("✅ Image embedding model and transforms loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit() # Exit if model cannot be loaded

# --- 2. Connect to MongoDB and Fetch Data ---
print("\n--- 2. Connecting to MongoDB and Fetching Product Data ---")
mongo_client = None
mongo_collection = None
product_data_for_indexing = [] # Stores {'_id': CID, 'cloudinary_url': URL}
try:
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable is not set.")
    
    mongo_client = pymongo.MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB_NAME]
    mongo_collection = mongo_db[MONGO_COLLECTION_NAME]
    print("✅ Connected to MongoDB Atlas.")

    # Fetch only _id and cloudinary_url, only for documents with a cloudinary_url
    cursor = mongo_collection.find(
        {"cloudinary_url": {"$ne": None}},
        {"_id": 1, "cloudinary_url": 1}
    )
    
    for doc in tqdm(cursor, desc="Fetching data from MongoDB"):
        product_data_for_indexing.append(doc)

    print(f"✅ Fetched {len(product_data_for_indexing)} products from MongoDB.")

except Exception as e:
    print(f"❌ Error connecting to MongoDB or fetching data: {e}")
    if mongo_client:
        mongo_client.close()
    exit()

if not product_data_for_indexing:
    print("No products found with Cloudinary URLs. Exiting.")
    if mongo_client:
        mongo_client.close()
    exit()

# --- 3. Generate Embeddings and Build FAISS Index ---
print("\n--- 3. Generating Embeddings and Building FAISS Index ---")

# Determine embedding dimension (output size of your model)
# Pass a dummy tensor through the model to get the dimension
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    embedding_dim = model(dummy_input).flatten().shape[0]

print(f"Embedding dimension: {embedding_dim}")

# Initialize FAISS index
# IndexFlatL2 is suitable for L2 (Euclidean) distance, common for embeddings
index = faiss.IndexFlatL2(embedding_dim)

all_embeddings = []
cids_in_order = []
failed_downloads = 0

for product_doc in tqdm(product_data_for_indexing, desc="Processing Images and Generating Embeddings"):
    cid = str(product_doc.get('_id'))
    cloudinary_url = product_doc.get('cloudinary_url')

    if not cloudinary_url:
        continue

    try:
        # Download image
        response = requests.get(cloudinary_url, timeout=10) # 10-second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Open image with PIL
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Apply transformations and get embedding
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0) # Add batch dimension
        batch_t = batch_t.to(device) # Move tensor to device

        with torch.no_grad():
            embedding = model(batch_t).cpu().numpy().flatten().astype('float32') # Get embedding

        all_embeddings.append(embedding)
        cids_in_order.append(cid)

    except requests.exceptions.Timeout:
        print(f"Timeout downloading image from {cloudinary_url}")
        failed_downloads += 1
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {cloudinary_url}: {e}")
        failed_downloads += 1
    except Exception as e:
        print(f"Error processing image {cloudinary_url}: {e}")
        failed_downloads += 1

if not all_embeddings:
    print("No embeddings generated. Index will not be created. Check image URLs or processing errors.")
    if mongo_client:
        mongo_client.close()
    exit()

# Convert list of embeddings to a single NumPy array
embeddings_array = np.array(all_embeddings)

# Add embeddings to the FAISS index
index.add(embeddings_array)
print(f"✅ FAISS index built. Total vectors added: {index.ntotal}")
if failed_downloads > 0:
    print(f"⚠️ Warning: {failed_downloads} images failed to download or process.")


# --- 4. Save FAISS Index and CID Map ---
print("\n--- 4. Saving FAISS Index and CID Map ---")
try:
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(CID_MAP_FILE, np.array(cids_in_order))
    print(f"✅ FAISS index saved to {FAISS_INDEX_FILE}")
    print(f"✅ CID map saved to {CID_MAP_FILE}")
except Exception as e:
    print(f"❌ Error saving files: {e}")

# --- Clean Up ---
if mongo_client:
    mongo_client.close()
    print("🔒 MongoDB connection closed.")

print("\n--- Index Creation Complete ---")