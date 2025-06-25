from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import uuid
import numpy as np
import base64
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial import distance
import pickle
import traceback
import re
from datetime import datetime, date
import pytesseract

# OCR and text processing imports
try:
    import pytesseract
    # You may need to set the path to tesseract executable
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
    OCR_AVAILABLE = True
    print("Tesseract OCR is available")
except ImportError:
    print("Tesseract OCR not available. Install with: pip install pytesseract")
    OCR_AVAILABLE = False

# FaceNet PyTorch imports
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    from torchvision import transforms
    FACENET_AVAILABLE = True
    print("FaceNet PyTorch is available")
except ImportError:
    print("FaceNet PyTorch not available. Install with: pip install facenet-pytorch")
    FACENET_AVAILABLE = False

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global variables to store the FaceNet model and MTCNN
facenet_model = None
mtcnn = None

def load_facenet_model():
    """Load the FaceNet model using facenet-pytorch"""
    global facenet_model, mtcnn
    
    if not FACENET_AVAILABLE:
        print("FaceNet PyTorch not available, using fallback model")
        facenet_model = create_simple_embedding_model()
        return
    
    try:
        # Initialize MTCNN for face detection (more accurate than Haar cascades)
        mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
            factor=0.709, 
            post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load pre-trained FaceNet model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print(f"FaceNet model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading FaceNet model: {e}")
        facenet_model = create_simple_embedding_model()
        mtcnn = None

def create_simple_embedding_model():
    """Create a simple CNN model as fallback if FaceNet is not available"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(160, 160, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128)  # Embedding layer
    ])
    return model

def preprocess_image_for_ocr(img):
    """Preprocess image for better OCR results"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Scale up image for better OCR
        scaled = cv2.resize(cleaned, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        return scaled
    except Exception as e:
        print(f"Error preprocessing image for OCR: {e}")
        return img

def extract_text_from_aadhaar(img):
    """Extract text from Aadhaar card using OCR"""
    if not OCR_AVAILABLE:
        return {"raw_text": "", "error": "OCR not available"}
    
    try:
        # Preprocess image for better OCR
        processed_img = preprocess_image_for_ocr(img)
        
        # Configure tesseract for better results
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./-: '
        
        # Extract text
        raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        print(f"Raw OCR text: {raw_text}")
        
        return {"raw_text": raw_text, "error": None}
    
    except Exception as e:
        print(f"Error extracting text from Aadhaar: {e}")
        traceback.print_exc()
        return {"raw_text": "", "error": str(e)}

def parse_aadhaar_info(raw_text):
    """Parse Aadhaar information from raw OCR text"""
    info = {
        "name": None,
        "dob": None,
        "age": None,
        "gender": None,
        "aadhaar_number": None,
        "father_name": None,
        "address": None
    }
    
    try:
        # Clean up text
        text = raw_text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove extra spaces
        
        # Extract Date of Birth
        dob_patterns = [
            r'DOB[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'Date of Birth[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'(\d{2}[/-]\d{2}[/-]\d{4})',
            r'DOB[:\s]*(\d{2}\.\d{2}\.\d{4})',
            r'(\d{2}\.\d{2}\.\d{4})'
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dob_str = match.group(1)
                info["dob"] = dob_str
                
                # Calculate age
                age = calculate_age_from_dob(dob_str)
                if age:
                    info["age"] = age
                break
        
        # Extract Name (usually appears before DOB or after "Name:")
        name_patterns = [
            r'Name[:\s]*([A-Za-z\s]+?)(?=DOB|Date of Birth|\d{2}[/-]\d{2}[/-]\d{4})',
            r'^([A-Z][A-Za-z\s]+?)(?=\s+\d{2}[/-]\d{2}[/-]\d{4})',
            r'Name[:\s]*([A-Z][A-Za-z\s]+?)(?=\s+(?:Male|Female|DOB))'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and len(name) < 50:  # Reasonable name length
                    info["name"] = name
                    break
        
        # Extract Gender
        gender_match = re.search(r'\b(Male|Female|M|F)\b', text, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group(1).upper()
            if gender in ['M', 'MALE']:
                info["gender"] = "Male"
            elif gender in ['F', 'FEMALE']:
                info["gender"] = "Female"
        
        # Extract Aadhaar Number (12 digits)
        aadhaar_patterns = [
            r'\b(\d{4}\s*\d{4}\s*\d{4})\b',
            r'\b(\d{12})\b'
        ]
        
        for pattern in aadhaar_patterns:
            match = re.search(pattern, text)
            if match:
                aadhaar = match.group(1).replace(' ', '')
                if len(aadhaar) == 12:
                    info["aadhaar_number"] = aadhaar
                    break
        
        # Extract Father's Name (if present)
        father_patterns = [
            r'Father[:\s]*([A-Za-z\s]+?)(?=\s+(?:DOB|Address|\d{2}[/-]\d{2}[/-]\d{4}))',
            r'S/O[:\s]*([A-Za-z\s]+?)(?=\s+(?:DOB|Address|\d{2}[/-]\d{2}[/-]\d{4}))'
        ]
        
        for pattern in father_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                father_name = match.group(1).strip()
                if len(father_name) > 2 and len(father_name) < 50:
                    info["father_name"] = father_name
                    break
        
        print(f"Parsed Aadhaar info: {info}")
        
    except Exception as e:
        print(f"Error parsing Aadhaar info: {e}")
        traceback.print_exc()
    
    return info

def calculate_age_from_dob(dob_str):
    """Calculate age from date of birth string"""
    try:
        # Try different date formats
        date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%m/%d/%Y', '%m-%d-%Y']
        
        birth_date = None
        for fmt in date_formats:
            try:
                birth_date = datetime.strptime(dob_str, fmt).date()
                break
            except ValueError:
                continue
        
        if birth_date:
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            # Sanity check for age
            if 0 <= age <= 150:
                return age
        
        return None
    
    except Exception as e:
        print(f"Error calculating age: {e}")
        return None

def detect_and_extract_face_mtcnn(img):
    """Detect and extract face using MTCNN (more accurate)"""
    global mtcnn
    
    if mtcnn is None:
        return detect_and_extract_face_opencv(img)
    
    try:
        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Detect faces and get bounding boxes
        boxes, _ = mtcnn.detect(pil_img)
        
        if boxes is not None and len(boxes) > 0:
            # Get the first detected face
            box = boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Extract face region
            face_img = img[y1:y2, x1:x2]
            
            # Resize to 160x160 for consistency
            face_img = cv2.resize(face_img, (160, 160))
            
            return face_img, (x1, y1, x2-x1, y2-y1)
        else:
            return None, None
            
    except Exception as e:
        print(f"Error with MTCNN face detection: {e}")
        traceback.print_exc()
        return detect_and_extract_face_opencv(img)

def detect_and_extract_face_opencv(img):
    """Fallback face detection using OpenCV Haar cascades"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
        
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
        face_img = img[y:y+h, x:x+w]
        
        # Resize to 160x160 for consistency
        face_img = cv2.resize(face_img, (160, 160))
        
        return face_img, (x, y, w, h)
    except Exception as e:
        print(f"Error with OpenCV face detection: {e}")
        traceback.print_exc()
        return None, None

def detect_and_extract_face(img):
    """Main face detection function that tries MTCNN first, then OpenCV"""
    if FACENET_AVAILABLE and mtcnn is not None:
        return detect_and_extract_face_mtcnn(img)
    else:
        return detect_and_extract_face_opencv(img)

def preprocess_face_for_facenet_pytorch(face_img):
    """Preprocess face image for FaceNet PyTorch model"""
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(face_rgb)
    
    # Resize to 160x160
    pil_img = pil_img.resize((160, 160))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    face_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
    
    return face_tensor

def preprocess_face_for_facenet_keras(face_img):
    """Preprocess face image for FaceNet Keras model (fallback)"""
    # Resize to 160x160 (FaceNet input size)
    face_resized = cv2.resize(face_img, (160, 160))
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    face_normalized = face_rgb.astype('float32') / 255.0
    
    # Add batch dimension
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def extract_face_embedding(face_img):
    """Extract face embedding using FaceNet"""
    global facenet_model
    
    if facenet_model is None:
        load_facenet_model()
    
    try:
        if FACENET_AVAILABLE and isinstance(facenet_model, torch.nn.Module):
            # Use FaceNet PyTorch
            face_tensor = preprocess_face_for_facenet_pytorch(face_img)
            
            device = next(facenet_model.parameters()).device
            face_tensor = face_tensor.to(device)
            
            with torch.no_grad():
                embedding = facenet_model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
        else:
            # Use Keras fallback model
            face_processed = preprocess_face_for_facenet_keras(face_img)
            embedding = facenet_model.predict(face_processed, verbose=0)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.flatten()
    
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        traceback.print_exc()
        # Fallback to simple features if everything fails
        return extract_simple_features(face_img)

def extract_simple_features(face_img):
    """Fallback feature extraction method"""
    face_resized = cv2.resize(face_img, (100, 100))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Simple histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = cv2.normalize(hist, hist).flatten()
    
    return hist_normalized

def calculate_face_similarity(embedding1, embedding2):
    """Calculate similarity between two face embeddings"""
    try:
        # Ensure embeddings are numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Ensure same dimensions
        min_len = min(len(emb1), len(emb2))
        emb1 = emb1[:min_len]
        emb2 = emb2[:min_len]
        
        # Calculate cosine similarity
        cosine_sim = 1 - distance.cosine(emb1, emb2)
        
        # Handle NaN values
        if np.isnan(cosine_sim):
            cosine_sim = 0
        
        # For FaceNet, cosine similarity is typically the primary metric
        if FACENET_AVAILABLE and isinstance(facenet_model, torch.nn.Module):
            # Use cosine similarity as primary metric for FaceNet
            similarity_percentage = max(0, min(100, cosine_sim * 100))
        else:
            # Calculate Euclidean distance similarity for fallback
            euclidean_dist = distance.euclidean(emb1, emb2)
            max_dist = np.sqrt(len(emb1))
            euclidean_sim = 1 - (euclidean_dist / max_dist)
            
            # Combined similarities
            combined_similarity = (cosine_sim * 0.7 + euclidean_sim * 0.3)
            similarity_percentage = max(0, min(100, combined_similarity * 100))
        
        return float(similarity_percentage)
    
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        traceback.print_exc()
        return 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'aadhaar' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['aadhaar']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Generate unique filename
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        print(f"File saved to: {file_path}")
        
        # Read and process image
        img = cv2.imread(file_path)
        if img is None:
            os.remove(file_path)
            return jsonify({'success': False, 'message': 'Invalid image file'})
        
        print(f"Image loaded successfully, shape: {img.shape}")
        
        # Extract text information from Aadhaar card
        ocr_result = extract_text_from_aadhaar(img)
        aadhaar_info = parse_aadhaar_info(ocr_result["raw_text"])
        
        # Detect and extract face
        face_img, face_coords = detect_and_extract_face(img)
        
        if face_img is None:
            os.remove(file_path)
            return jsonify({'success': False, 'message': 'No face detected in the uploaded image'})
        
        print(f"Face detected, coordinates: {face_coords}")
        
        # Save extracted face
        face_filename = f"{uuid.uuid4().hex}-face.jpg"
        face_path = os.path.join(UPLOAD_FOLDER, face_filename)
        
        # Ensure the face image is saved correctly
        success = cv2.imwrite(face_path, face_img)
        if not success:
            os.remove(file_path)
            return jsonify({'success': False, 'message': 'Failed to save extracted face'})
        
        print(f"Face saved to: {face_path}")
        
        # Extract face embedding using FaceNet
        face_embedding = extract_face_embedding(face_img)
        
        if face_embedding is None:
            os.remove(file_path)
            os.remove(face_path)
            return jsonify({'success': False, 'message': 'Failed to extract face embedding'})
        
        print(f"Face embedding extracted, shape: {face_embedding.shape}")
        
        # Save embedding
        embedding_filename = f"{uuid.uuid4().hex}-embedding.pkl"
        embedding_path = os.path.join(UPLOAD_FOLDER, embedding_filename)
        with open(embedding_path, 'wb') as f:
            pickle.dump(face_embedding, f)
        
        print(f"Embedding saved to: {embedding_path}")
        
        # Save Aadhaar info
        info_filename = f"{uuid.uuid4().hex}-info.pkl"
        info_path = os.path.join(UPLOAD_FOLDER, info_filename)
        with open(info_path, 'wb') as f:
            pickle.dump(aadhaar_info, f)
        
        return jsonify({
            'success': True, 
            'face_url': f'/uploads/{face_filename}',
            'embedding_file': embedding_filename,
            'info_file': info_filename,
            'aadhaar_info': aadhaar_info,
            'ocr_available': OCR_AVAILABLE,
            'message': f'Face detected and processed successfully'
        })
    
    except Exception as e:
        print(f"Error in upload: {e}")
        traceback.print_exc()
        # Clean up files if they exist
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        if 'face_path' in locals() and os.path.exists(face_path):
            os.remove(face_path)
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})

@app.route('/capture_live', methods=['POST'])
def capture_live():
    try:
        # Get the base64 image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, part
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        print(f"Live image captured, shape: {opencv_image.shape}")
        
        # Detect and extract face
        live_face, face_coords = detect_and_extract_face(opencv_image)
        
        if live_face is None:
            return jsonify({'success': False, 'message': 'No face detected in live image'})
        
        print(f"Live face detected, coordinates: {face_coords}")
        
        # Save live face image
        live_face_filename = f"{uuid.uuid4().hex}-live.jpg"
        live_face_path = os.path.join(UPLOAD_FOLDER, live_face_filename)
        
        success = cv2.imwrite(live_face_path, live_face)
        if not success:
            return jsonify({'success': False, 'message': 'Failed to save live face'})
        
        print(f"Live face saved to: {live_face_path}")
        
        # Extract face embedding using FaceNet
        live_embedding = extract_face_embedding(live_face)
        
        if live_embedding is None:
            os.remove(live_face_path)
            return jsonify({'success': False, 'message': 'Failed to extract live face embedding'})
        
        print(f"Live embedding extracted, shape: {live_embedding.shape}")
        
        return jsonify({
            'success': True,
            'live_face_url': f'/uploads/{live_face_filename}',
            'live_embedding': live_embedding.tolist(),  # Convert to list for JSON
            'message': 'Live face captured and processed successfully'
        })
        
    except Exception as e:
        print(f"Error in capture_live: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error processing live image: {str(e)}'})

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        embedding_file = data.get('embedding_file')
        info_file = data.get('info_file')
        live_embedding = data.get('live_embedding')
        
        print(f"Comparing faces - embedding_file: {embedding_file}, info_file: {info_file}, live_embedding length: {len(live_embedding) if live_embedding else 'None'}")
        
        if not embedding_file:
            return jsonify({'success': False, 'message': 'No reference embedding file provided'})
        
        if not live_embedding:
            return jsonify({'success': False, 'message': 'No live embedding provided'})
        
        # Load reference embedding
        embedding_path = os.path.join(UPLOAD_FOLDER, embedding_file)
        if not os.path.exists(embedding_path):
            return jsonify({'success': False, 'message': f'Reference embedding file not found: {embedding_file}'})
        
        with open(embedding_path, 'rb') as f:
            reference_embedding = pickle.load(f)
        
        print(f"Reference embedding loaded, shape: {reference_embedding.shape}")
        
        # Load Aadhaar info if available
        aadhaar_info = {}
        if info_file:
            info_path = os.path.join(UPLOAD_FOLDER, info_file)
            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    aadhaar_info = pickle.load(f)
                print(f"Aadhaar info loaded: {aadhaar_info}")
        
        # Convert live embedding to numpy array
        live_embedding_array = np.array(live_embedding)
        print(f"Live embedding converted, shape: {live_embedding_array.shape}")
        
        # Calculate similarity using FaceNet embeddings
        similarity_score = calculate_face_similarity(reference_embedding, live_embedding_array)
        
        print(f"Similarity calculated: {similarity_score}")
        
        # Determine match status with appropriate thresholds
        if FACENET_AVAILABLE and isinstance(facenet_model, torch.nn.Module):
            threshold = 70.0  # FaceNet typically uses lower thresholds due to better accuracy
            method_used = 'FaceNet PyTorch (VGGFace2)'
        else:
            threshold = 75.0  # Higher threshold for fallback methods
            method_used = 'Fallback CNN + Traditional Features'
            
        is_match = bool(similarity_score >= threshold)
        
        # Ensure all values are JSON serializable
        similarity_score = float(similarity_score)
        threshold = float(threshold)
        
        # Generate analytics with Aadhaar information
        analytics = {
            'similarity_percentage': similarity_score,
            'is_match': is_match,
            'confidence_level': 'High' if similarity_score >= 85 else 'Medium' if similarity_score >= 70 else 'Low',
            'match_status': 'Match' if is_match else 'No Match',
            'threshold_used': threshold,
            'method_used': str(method_used),
            'using_facenet': bool(FACENET_AVAILABLE and isinstance(facenet_model, torch.nn.Module)),
            'aadhaar_info': aadhaar_info,
            'person_age': aadhaar_info.get('age'),
            'person_name': aadhaar_info.get('name'),
            'person_dob': aadhaar_info.get('dob'),
            'person_gender': aadhaar_info.get('gender')
        }
        
        print(f"Analytics generated: {analytics}")
        
        return jsonify({
            'success': True,
            'analytics': analytics
        })
        
    except Exception as e:
        print(f"Error in compare_faces: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error comparing faces: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/status')
def status():
    """Endpoint to check model status"""
    return jsonify({
        'facenet_available': FACENET_AVAILABLE,
        'using_pytorch': isinstance(facenet_model, torch.nn.Module) if facenet_model else False,
        'mtcnn_available': mtcnn is not None,
        'cuda_available': torch.cuda.is_available() if FACENET_AVAILABLE else False,
        'ocr_available': OCR_AVAILABLE
    })

if __name__ == '__main__':
    # Load model at startup
    print("Loading FaceNet model...")
    load_facenet_model()
    print("Model loaded. Starting Flask app...")
    app.run(debug=True)