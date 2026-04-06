import cv2
import numpy as np
import os
import gradio as gr
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Core Image Processing Pipeline (Based on the Paper)
# ---------------------------------------------------------

def extract_main_content(image):
    """
    Pre-processes and segments the image using Median Filtering, 
    Otsu's Method on Saturation, and Contour Hole-Filling.
    """
    if image is None:
        return None, None
        
    # 1. Apply median filter to remove noise
    filtered_image = cv2.medianBlur(image, 5)
    
    # 2. Convert RGB to HSV and get Saturation
    hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2HSV)
    saturation_channel = hsv_image[:, :, 1] 
    
    # 3. Apply Otsu's thresholding
    _, threshold_mask = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Failsafe: Invert mask if background was selected
    if threshold_mask[0, 0] == 255:
        threshold_mask = cv2.bitwise_not(threshold_mask)
        
    # 5. NEW FIX: Fill the holes caused by mold/rot
    # Find the boundaries (contours) of all detected objects
    contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour is our fruit
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a fresh, all-black mask
        filled_mask = np.zeros_like(threshold_mask)
        
        # Draw the largest contour and fill it completely solid (thickness=-1)
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Replace the old holey mask with our new solid mask
        threshold_mask = filled_mask
    
    # 6. Extract the main content by applying the solid mask
    roi_image = cv2.bitwise_and(image, image, mask=threshold_mask)
    
    return roi_image, threshold_mask
def extract_features(roi_image):
    """
    Extracts statistical and texture features from the segmented ROI.
    """
    # Convert to grayscale for texture analysis
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
    
    # 1. Statistical Features [cite: 129]
    mean_val = np.mean(gray_roi)
    std_val = np.std(gray_roi)
    
    # 2. Texture Features using GLCM [cite: 130]
    glcm = graycomatrix(gray_roi, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [mean_val, std_val, contrast, energy, homogeneity, correlation]

# ---------------------------------------------------------
# 2. Model Training Phase
# ---------------------------------------------------------

def train_model(dataset_path):
    """
    Iterates through the dataset, extracts features, and trains a KNN Classifier.
    """
    print(f"Loading dataset from: {dataset_path}")
    X_train = []
    y_train = []
    
    # Assumes the archive folder has subfolders for each class (e.g., Apple, Banana)
    if not os.path.exists(dataset_path):
        return None, None, "Dataset path not found. Please verify the path."

    classes = os.listdir(dataset_path)
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run pipeline
            roi_image, _ = extract_main_content(image)
            features = extract_features(roi_image)
            
            X_train.append(features)
            y_train.append(class_name)
            
    if len(X_train) == 0:
        return None, None, "No images found to train."

    # Scale features and Train KNN Classifier [cite: 203, 207]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_scaled, y_train)
    
    return knn_model, scaler, f"Model successfully trained on {len(X_train)} images across {len(classes)} classes."

# ---------------------------------------------------------
# 3. Prediction and Interface Integration
# ---------------------------------------------------------

DATASET_PATH = r"C:\Users\teja6\Desktop\mfc-project\archive"
model, feature_scaler, status_message = train_model(DATASET_PATH)
print(status_message)

def process_input(input_image):
    if model is None or feature_scaler is None:
        return None, "Error: Model not trained. Check dataset path."
        
    if input_image is None:
        return None, "Please provide an image."
        
    # 1. Segment the image
    roi_image, _ = extract_main_content(input_image)
    
    # 2. Extract features
    features = extract_features(roi_image)
    
    # 3. Predict using the trained model
    scaled_features = feature_scaler.transform([features])
    
    # Calculate probabilities across all classes
    probabilities = model.predict_proba(scaled_features)[0]
    
    # Aggregate probabilities into Condition (Healthy vs. Spoiled)
    prob_healthy = 0.0
    prob_spoiled = 0.0
    
    for cls, prob in zip(model.classes_, probabilities):
        if "fresh" in cls.lower():
            prob_healthy += prob
        elif "stale" in cls.lower() or "damaged" in cls.lower() or "rotten" in cls.lower():
            prob_spoiled += prob
            
    # Determine the final condition and confidence
    if prob_healthy >= prob_spoiled:
        final_condition = "Healthy"
        confidence = prob_healthy * 100
    else:
        final_condition = "Spoiled"
        confidence = prob_spoiled * 100
    
    return roi_image, f"Condition: {final_condition} ({confidence:.2f}%)"

# ---------------------------------------------------------
# 4. Gradio Interface (Drag-and-Drop & Camera)
# ---------------------------------------------------------

interface = gr.Interface(
    fn=process_input,
    inputs=gr.Image(sources=["upload", "webcam"], type="numpy", label="Input Image (Drag & Drop or Camera)"),
    outputs=[
        gr.Image(type="numpy", label="Extracted Main Content (ROI)"),
        gr.Textbox(label="Condition (Healthy / Spoiled)")
    ],
    title="Image Processing & Classification Pipeline",
    description="Based on 'A STUDY ON VARIOUS IMAGE PROCESSING TECHNIQUES'."
)

if __name__ == "__main__":
    interface.launch()