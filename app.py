# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image
import os # For environment variables
from dotenv import load_dotenv # For loading .env file

# Load environment variables from .env file (if it exists)
load_dotenv()

# Import your similarity search module
import similarity_module as sm

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Global Variables for Loaded Resources (Managed by similarity_module) ---
# These are managed within similarity_module, so we don't need to declare them global here.

# --- Resource Initialization Function ---
# This function will now be called explicitly when the app starts.
# It no longer needs to be decorated with @app.before_first_request.
def initialize_resources():
    """
    This function initializes heavy resources like the FAISS index and ML model.
    It's designed to be called once when the application starts.
    """
    print("Initializing resources for Flask app via similarity_module...")
    sm.load_faiss_index_and_map() # Load FAISS index and CID map
    sm.load_image_embedding_model_and_transform() # Load model and transforms
    sm.get_mongo_collection() # Establish MongoDB connection

    # Check if all resources are loaded
    if sm._faiss_index is None or \
       sm._image_embedding_model is None or \
       sm._mongo_collection is None or \
       sm._faiss_index_to_cid_map is None:
        print("ERROR: One or more critical resources failed to load. API might not function correctly.")
        # In a production app, you might want to raise an exception or exit here
        # For now, it will just log and try to proceed.

@app.route('/visual_search', methods=['POST'])
def visual_search():
    # Check if an image file was provided in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # 1. Read the image and convert to PIL Image object
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # 2. Get embedding of the query image
            query_embedding = sm.get_image_embedding(image)

            # 3. Perform similarity search using FAISS
            k_results = request.args.get('k', 5, type=int) # Allow 'k' to be passed as query param, default 5
            faiss_results = sm.search_faiss(query_embedding, k=k_results)

            # Extract CIDs from FAISS results
            matched_cids = [res['cid'] for res in faiss_results]

            # 4. Fetch detailed product information from MongoDB
            product_details = sm.get_product_details_from_mongo(matched_cids)

            # Ensure the results are sorted by distance from FAISS
            cid_distance_map = {res['cid']: res['distance'] for res in faiss_results}
            
            for pd in product_details:
                pd['distance'] = cid_distance_map.get(pd['CID'], float('inf')) 
            
            sorted_product_details = sorted(product_details, key=lambda x: x['distance'])


            # Return the results as JSON
            return jsonify({"results": sorted_product_details}), 200

        except ValueError as ve:
            print(f"Configuration/Resource error: {str(ve)}")
            return jsonify({"error": f"Configuration error: {str(ve)}"}), 500
        except Exception as e:
            print(f"Error processing visual search request: {e}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Main entry point for running the Flask app
if __name__ == '__main__':
    # Call initialize_resources() explicitly here when the script is run directly.
    # We wrap it in app.app_context() to ensure Flask's application context is available.
    with app.app_context():
        initialize_resources()
        
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True for development, set to False for production!