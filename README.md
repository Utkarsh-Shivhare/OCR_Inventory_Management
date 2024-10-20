# OCR Inventory Manager

This project demonstrates how to manage inventory and create bills using an OCR-based system integrated with Flask. The app automates the process of reading product details from images, extracting text using OCR, and matching those details against a product inventory stored in an Excel file.

## Overview
Manually feeding product details into a system can be time-consuming and error-prone. This project provides an automated solution to manage inventory using OCR (Optical Character Recognition) technology. By uploading product images, the app extracts text data and matches it against a pre-existing inventory, helping businesses generate bills and track inventory automatically.

### Features
- **OCR Integration**: Extracts text from product images.
- **Product Matching**: Matches OCR output with product inventory using fuzzy matching and FAISS search for fast and accurate results.
- **Multi-Image Upload**: Supports uploading multiple images for batch processing.
- **Flask Web Interface**: Provides a simple web interface to upload images and view detected products.

## How to Set Up the Application

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd OCR_INVENTORY_MANAGER
   ```

2. **Install Python and Virtual Environment (Optional)**:
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install Dependencies**: Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Inventory**: The app requires a product inventory in the form of an Excel file (Product_Inventory.xlsx). Ensure your Excel file has the following structure:
   - **Brand**: The brand of the product.
   - **Product Name**: The name of the product.

   You can either use the provided Product_Inventory.xlsx or create your own file with the same format.

5. **Run the Application**: Start the Flask server by running:
   ```bash
   python app.py
   ```
   The server will start at [http://localhost:5000/](http://localhost:5000/). Open this link in your browser.

## Using the App
- Upload product images via the web interface.
- The app will extract text from the images using OCR and attempt to match the extracted text with products in the inventory.
- You will see the detected products on the result page.