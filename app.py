from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from paddleocr import PaddleOCR
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from concurrent.futures import ThreadPoolExecutor

# Initialize the Flask app
app = Flask(__name__)

# Load product data and preprocess it
file_path = 'Product_Inventory.xlsx'
df = pd.read_excel(file_path)

# Normalize the product names and brands to lowercase
df['Brand'] = df['Brand'].str.lower()
df['Product Name'] = df['Product Name'].str.lower()

# Extract unique brands for optimization
unique_brands = df['Brand'].unique()

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Helper function: OCR extraction
def ocr_image_text(image_path):
    ocr_result = ocr.ocr(image_path, cls=True)
    result_text = ""
    for idx in range(len(ocr_result)):
        for line in ocr_result[idx]:
            result_text += line[1][0] + " "
    return result_text.lower()

# Helper function: Fuzzy matching
def fuzzy_match(text, candidates):
    return process.extractOne(text, candidates)[0]

# Helper function: Preprocess product for matching
def preprocess_product(product_row):
    product_name = ' '.join(str(product_row['Product Name']).split()[0:])  # Remove the first word
    return f"{product_name}"

# Step 5: Use TF-IDF for vectorizing product strings
vectorizer = TfidfVectorizer()
product_strings = df.apply(preprocess_product, axis=1).tolist()
product_vectors = vectorizer.fit_transform(product_strings).toarray()

# FAISS index for fast search
d = product_vectors.shape[1]
faiss_index = faiss.IndexFlatL2(d)  # Renamed from 'index' to 'faiss_index'
faiss_index.add(product_vectors)

# Function to search for top matches using FAISS
def search_with_faiss(query, faiss_index, vectorizer, top_k=3):  # Rename 'index' to 'faiss_index'
    query_vector = vectorizer.transform([query]).toarray()
    D, I = faiss_index.search(query_vector, top_k)  # Use 'faiss_index' instead of 'index'
    return I[0]  # Return indices of the matches

# Function to refine FAISS results using fuzzy matching
# Function to refine FAISS results using fuzzy matching
def fuzzy_refine_match(product_index, query_string, product_strings):
    # Ensure that the product index is valid
    if product_index < len(product_strings):
        product_string = product_strings[product_index]
        fuzzy_score = process.extractOne(query_string, [product_string])[1]
        return product_string, fuzzy_score
    else:
        # Return a low score for invalid indices
        return None, 0

# Function to handle multiple matches using threads
def parallel_fuzzy_refine(matches, query_string, product_strings, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(fuzzy_refine_match, match, query_string, product_strings) 
            for match in matches if match < len(product_strings)  # Ensure valid index
        ]
        results = [future.result() for future in futures if future.result()[0]]  # Filter out None results
    return sorted(results, key=lambda x: x[1], reverse=True)

# Function to search for top matches using FAISS
def search_with_faiss(query, faiss_index, vectorizer, product_strings, top_k=3):
    query_vector = vectorizer.transform([query]).toarray()
    D, I = faiss_index.search(query_vector, top_k)
    
    # Ensure indices are within valid range
    valid_matches = [i for i in I[0] if i < len(product_strings)]
    
    return valid_matches

@app.route('/', methods=['GET', 'POST'])
def index():
    # Clear the uploads directory at the start of the function
    upload_folder = 'uploads'
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    if request.method == 'POST':
        files = request.files.getlist('images')
        detected_products = {}

        for file in files:
            # Save the uploaded file temporarily
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)

            # Perform OCR on the image
            ocr_text = ocr_image_text(image_path)
            print(ocr_text)
            # Fuzzy match the brand from the OCR text
            brand = fuzzy_match(ocr_text, unique_brands)

            # Filter products based on the matched brand
            brand_products = df[df['Brand'] == brand]

            # Prepare product strings for matching
            product_strings = brand_products.apply(preprocess_product, axis=1).tolist()

            # Fuzzy match the product name
            product_name = fuzzy_match(ocr_text, product_strings)

            # Define query string
            query_string = f"{brand} {product_name}"
            print(query_string)
            # Get top FAISS matches
            # top_matches = search_with_faiss(query_string, faiss_index, vectorizer, product_strings)

            # # Refine the top FAISS matches
            # refined_matches = parallel_fuzzy_refine(top_matches, query_string, product_strings)

            # # Get the best product
            # best_product = refined_matches[0][0] if refined_matches else None

            # Add the result to the dictionary
            # detected_products[file.filename] = f"{brand.title()} {product_name.title()}" 
            product_key = f"{brand.title()} {product_name.title()}"
            if product_key in detected_products:
                detected_products[product_key]["count"] += 1
            else:
                detected_products[product_key] = {
                    "product": product_key,
                    "count": 1,
                    "file_name": file.filename  # Store the file name for frontend use
                }

        # Return the results as a dictionary
        return jsonify(detected_products)

    return render_template('index.html')

# Start the Flask app
if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
