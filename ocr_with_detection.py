from ultralytics import YOLO
from paddleocr import PaddleOCR
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from fuzzywuzzy import process
from concurrent.futures import ThreadPoolExecutor
import os

# Load YOLO model for object detection
model = YOLO('Item_count_yolov8.pt')  # Path to your YOLOv8 model

# Load product data from the Excel file
file_path = 'Product_Inventory.xlsx'
df = pd.read_excel(file_path)

# Normalize the product names and brands to lowercase
df['Brand'] = df['Brand'].str.lower()
df['Product Name'] = df['Product Name'].str.lower()

# Extract unique brands for fuzzy matching
unique_brands = df['Brand'].unique()

# OCR Function to extract text from detected items
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize OCR for English

def ocr_image_text(image_path):
    ocr_result = ocr.ocr(image_path, cls=True)
    result_text = ""
    for idx in range(len(ocr_result)):
        for line in ocr_result[idx]:
            result_text += line[1][0] + " "
    return result_text.lower()

# Step 1: Detect items using YOLO
image_path = 'test_image_2.jpg'  # Place your image path here
results = model.predict(source=image_path, imgsz=1080, device='cpu', conf=0.25, save=True)

# Count the number of items detected
num_detected_items = len(results[0].boxes)  # 'boxes' holds the detected objects
print(f"Number of items detected: {num_detected_items}")

# Display the image with bounding boxes
for result in results:
    result.show()

# Process each detected object by extracting the portion of the image where the object is detected
detected_items_text = []

for idx, box in enumerate(results[0].boxes.xyxy):  # Get bounding box coordinates (x1, y1, x2, y2)
    cropped_image_path = f'detected_item_{idx}.jpg'
    
    # Crop the detected object from the image (using PIL)
    from PIL import Image
    img = Image.open(image_path)
    x1, y1, x2, y2 = map(int, box)
    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img.save(cropped_image_path)

    # Perform OCR on the cropped image
    detected_text = ocr_image_text(cropped_image_path)
    detected_items_text.append(detected_text)

    # Optionally, remove the cropped image file after processing
    os.remove(cropped_image_path)

# Now, match the OCR-detected text with the inventory

# Helper function: Fuzzy match the brand and product
def fuzzy_match(text, candidates):
    return process.extractOne(text, candidates)[0]

# Helper function: Preprocess product for matching
def preprocess_product(product_row):
    product_name = ' '.join(str(product_row['Product Name']).split()[0:])
    return f"{product_name}"

# Prepare for matching using TF-IDF and FAISS
vectorizer = TfidfVectorizer()
product_strings = df.apply(preprocess_product, axis=1).tolist()
product_vectors = vectorizer.fit_transform(product_strings).toarray()

# FAISS index for fast search
d = product_vectors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(product_vectors)

# Function to perform FAISS search
def search_with_faiss(query, index, vectorizer, top_k=3):
    query_vector = vectorizer.transform([query]).toarray()
    D, I = index.search(query_vector, top_k)  # Top k nearest matches
    return I[0]  # Return indices of the matches

# Function to refine matches using fuzzy matching
def fuzzy_refine_match(product_index, query_string):
    product_string = product_strings[product_index]
    fuzzy_score = process.extractOne(query_string, [product_string])[1]
    return product_string, fuzzy_score

# Use multithreading to refine the matches
def parallel_fuzzy_refine(matches, query_string, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(fuzzy_refine_match, match, query_string) for match in matches]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda x: x[1], reverse=True)

# Iterate over detected items and match them with the inventory
for detected_text in detected_items_text:
    print(f"Detected Text from OCR: {detected_text}")
    
    # Fuzzy match the brand from the OCR text
    brand = fuzzy_match(detected_text, unique_brands)
    print(f"Extracted Brand: {brand}")
    
    # Filter products based on the matched brand
    brand_products = df[df['Brand'] == brand]

    # Prepare product strings for matching
    product_strings = brand_products.apply(preprocess_product, axis=1).tolist()

    # Fuzzy match the product name
    product_name = fuzzy_match(detected_text, product_strings)
    print(f"Product Name: {product_name}")

    # Define query string for FAISS
    query_string = f"{product_name}"

    # # Get top FAISS matches
    # top_matches = search_with_faiss(query_string, index, vectorizer)

    # # Refine the top FAISS matches
    # refined_matches = parallel_fuzzy_refine(top_matches, query_string)

    # best_product = refined_matches[0][0] if refined_matches else None

    # Output the identified product
    if query_string:
        print(f"Identified Product: {brand.title()} {query_string.title()}")
    else:
        print("No matching product found.")
