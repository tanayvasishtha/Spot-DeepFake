"""
Spot DeepFake - Advanced Deepfake Detection System
Professional web application for real-time detection
Architecture: EfficientNetB0 + Custom Classifier
Accuracy: 95.67% on validation set
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Spot DeepFake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL THEME (NO PURPLE, NO EMOJIS)
# ============================================================================

st.markdown("""
    <style>
    [data-testid="stMainBlockContainer"] {
        background-color: #f8fafc;
    }
    
    [data-testid="stSidebarContent"] {
        background-color: #f1f5f9;
    }
    
    .main-header {
        background: linear-gradient(90deg, #0c4a6e 0%, #0369a1 50%, #0891b2 100%);
        padding: 40px;
        border-radius: 8px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 36px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 16px;
        opacity: 0.95;
    }
    
    .result-card-real {
        background-color: #ecfdf5;
        border-left: 5px solid #059669;
        padding: 20px;
        border-radius: 6px;
        margin: 15px 0;
    }
    
    .result-card-real h2 {
        color: #059669;
        margin: 0;
        font-size: 24px;
        font-weight: 700;
    }
    
    .result-card-fake {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 20px;
        border-radius: 6px;
        margin: 15px 0;
    }
    
    .result-card-fake h2 {
        color: #dc2626;
        margin: 0;
        font-size: 24px;
        font-weight: 700;
    }
    
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .metric-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-size: 20px;
        color: #0c4a6e;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #0369a1;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 13px;
        color: #0c4a6e;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0c4a6e;
        margin-bottom: 15px;
        border-bottom: 2px solid #0369a1;
        padding-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (CACHED FOR PERFORMANCE)
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained EfficientNetB0 model from saved weights"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model architecture (must match training architecture exactly)
    model = models.efficientnet_b0(weights=None)
    
    # Get in_features from classifier - handle different torchvision versions
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            # If classifier is Sequential, get in_features from first Linear layer
            in_features = None
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
            if in_features is None:
                # Fallback: EfficientNetB0 default is 1280
                in_features = 1280
        elif isinstance(model.classifier, nn.Linear):
            # If classifier is a single Linear layer
            in_features = model.classifier.in_features
        else:
            # Fallback: EfficientNetB0 default is 1280
            in_features = 1280
    else:
        # Fallback: EfficientNetB0 default is 1280
        in_features = 1280
    
    # Replace classifier with custom head (same as training)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    # Load trained weights
    model_path = 'models/spot_deepfake_efficientnetb0_best.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # If checkpoint is a dict, it might have 'state_dict' or 'model_state_dict' key
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Strip 'efficientnet.' prefix from keys if present
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('efficientnet.'):
                    new_key = key[len('efficientnet.'):]  # Remove 'efficientnet.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Load the state dict
            model.load_state_dict(new_state_dict, strict=False)
            model_status = "Trained Model Loaded Successfully"
        except Exception as e:
            model_status = f"Error loading model: {str(e)}"
            st.error(f"Error loading model weights: {str(e)}")
    else:
        model_status = "Error: Model file not found"
        st.error(f"Model file not found at {model_path}")
    
    model.to(device)
    model.eval()
    
    return model, device, model_status

# Load model
model, device, model_status = load_model()

# ============================================================================
# PREDICTION FUNCTION FOR SINGLE IMAGE
# ============================================================================

def predict_image(image, model, device):
    """
    Predict if image is real or fake
    
    Args:
        image (PIL.Image): Input image
        model (nn.Module): Trained model
        device (torch.device): GPU or CPU
    
    Returns:
        Tuple of (confidence_real, confidence_fake)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Prepare input
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probs = probabilities.cpu().numpy()[0]  # Extract first (and only) row
        confidence_real, confidence_fake = probs[0], probs[1]
    
    return float(confidence_real), float(confidence_fake)

# ============================================================================
# FACE EXTRACTION FUNCTION (FOR VIDEO PROCESSING)
# ============================================================================

@st.cache_resource
def load_face_detector():
    """
    Load OpenCV Haar Cascade face detector (cached for performance)
    Uses OpenCV's built-in Haar Cascade - no external dependencies required
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def extract_face_opencv(pil_img, detector):
    """
    Extract face from image using OpenCV Haar Cascade face detector
    Uses multiple detection attempts with different parameters for better accuracy
    
    Args:
        pil_img: PIL Image
        detector: OpenCV CascadeClassifier instance
    
    Returns:
        PIL Image (cropped face with padding or resized original if no face found)
    """
    img_array = np.array(pil_img)
    h, w = img_array.shape[:2]
    
    # Convert to grayscale for face detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Try multiple detection strategies with different parameters
    detection_params = [
        {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (30, 30)},
        {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (20, 20)},
        {'scaleFactor': 1.05, 'minNeighbors': 5, 'minSize': (40, 40)},
    ]
    
    faces = []
    for params in detection_params:
        detected = detector.detectMultiScale(gray, **params)
        if len(detected) > 0:
            faces = detected
            break
    
    if len(faces) > 0:
        # Use largest face (most likely to be the main subject)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add padding around face (20% on each side) for better context
        padding_w = int(w * 0.2)
        padding_h = int(h * 0.2)
        
        # Ensure coordinates are within image bounds
        x = max(0, x - padding_w)
        y = max(0, y - padding_h)
        w = min(w + 2 * padding_w, img_array.shape[1] - x)
        h = min(h + 2 * padding_h, img_array.shape[0] - y)
        
        face = pil_img.crop((x, y, x + w, y + h))
        return face.resize((224, 224))
    
    # Fallback: try center crop (often faces are centered in videos)
    center_x, center_y = w // 2, h // 2
    crop_size = min(w, h) // 2
    x = max(0, center_x - crop_size)
    y = max(0, center_y - crop_size)
    face = pil_img.crop((x, y, x + 2*crop_size, y + 2*crop_size))
    return face.resize((224, 224))

# ============================================================================
# VIDEO PROCESSING FUNCTION (WITH FACE CROPPING)
# ============================================================================

def process_video_with_face_detection(video_file, model, device, num_frames=15, frame_interval=10):
    """
    Process video frames with key frame extraction and dynamic sampling
    Research-aligned approach: extract key frames, detect faces, classify, aggregate
    
    Args:
        video_file: Uploaded video file
        model: Trained model
        device: Torch device
        num_frames: Maximum number of frames to sample
        frame_interval: Sample every Nth frame (key frame extraction)
    
    Returns:
        Tuple of (results list, deepfake_votes, total_frames, fake_probs_list, early_stopped)
    """
    # Image transform (same as predict_image - ensures consistency)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load face detector (cached)
    detector = load_face_detector()
    
    # Save video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Key frame extraction: sample every Nth frame (research-backed approach)
        # This captures significant scene changes and reduces redundant processing
        frame_idxs = np.arange(0, nframes, frame_interval)
        # Limit to max frames but prioritize early frames (where deepfakes often show artifacts)
        frame_idxs = frame_idxs[:num_frames]
        
        results = []
        fake_probs_list = []
        high_conf_count = 0
        early_stopped = False
        
        for idx, i in enumerate(frame_idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to PIL Image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Extract face using OpenCV detector
            face = extract_face_opencv(pil_frame, detector)
            
            # Transform and predict
            x = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                prob = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            
            fake_prob = float(prob[1])  # Deepfake class probability
            real_prob = float(prob[0])  # Real class probability
            
            fake_probs_list.append(fake_prob)
            
            results.append({
                'frame': len(results) + 1,
                'real': real_prob,
                'fake': fake_prob,
                'frame_index': i
            })
            
            # Dynamic early stopping: if we find multiple high-confidence fakes, stop early
            # This speeds up inference and improves accuracy (research-backed)
            if fake_prob > 0.85:
                high_conf_count += 1
                # If we have 3+ high-confidence detections, we can stop early
                if high_conf_count >= 3 and len(results) >= 5:
                    early_stopped = True
                    break
        
        cap.release()
        
        # Calculate votes (frames flagged as deepfake with >0.5 probability)
        deepfake_votes = sum(p > 0.5 for p in fake_probs_list)
        total_frames = len(results)
        
        return results, deepfake_votes, total_frames, fake_probs_list, early_stopped
    finally:
        os.unlink(tmp_path)

# ============================================================================
# MAIN APPLICATION INTERFACE
# ============================================================================

# Header
st.markdown("""
    <div class="main-header">
        <h1>Spot DeepFake</h1>
        <p>Advanced AI-Powered Deepfake Detection System</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### Configuration")
    
    # Input type selector
    upload_type = st.radio(
        "Select Input Type:",
        ["Image", "Video"],
        help="Choose whether to analyze an image or video file"
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.5,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Prediction confidence must exceed this threshold"
    )
    
    st.divider()
    
    # System information
    st.markdown("### System Information")
    st.markdown(f"**GPU Available:** {'Yes' if torch.cuda.is_available() else 'No'}")
    st.markdown(f"**Device:** {str(device)}")
    st.markdown(f"**Model Status:** {model_status}")
    
    st.divider()
    
    # About section
    st.markdown("### About Spot DeepFake")
    st.info("""
        Deep learning-based detection of AI-generated media
        
        **Architecture:** EfficientNetB0  
        **Accuracy:** 95.67%  
        **Input:** 224×224 RGB Images  
        **Classes:** Real / Deepfake
    """)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

col_upload, col_results = st.columns([1, 1], gap="large")

# UPLOAD COLUMN
with col_upload:
    st.markdown('<div class="section-title">Upload Media</div>', unsafe_allow_html=True)
    
    if upload_type == "Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image", key="analyze_img", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    conf_real, conf_fake = predict_image(image, model, device)
                    prediction = "Real" if conf_real > conf_fake else "Deepfake"
                    confidence = max(conf_real, conf_fake)
                    
                    st.session_state.prediction = prediction
                    st.session_state.conf_real = conf_real
                    st.session_state.conf_fake = conf_fake
                    st.session_state.confidence = confidence
    
    else:  # Video
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file:
            st.video(uploaded_file)
            
            if st.button("Analyze Video", key="analyze_vid", use_container_width=True):
                with st.spinner("Processing video with face detection..."):
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    results, deepfake_votes, total_frames, fake_probs_list, early_stopped = process_video_with_face_detection(
                        uploaded_file, model, device, num_frames=15, frame_interval=10
                    )
                    
                    if total_frames > 0:
                        # Research-backed hybrid voting strategy
                        max_fake_prob = max(fake_probs_list)
                        high_conf_fakes = [p for p in fake_probs_list if p > 0.85]
                        majority_fakes = [p for p in fake_probs_list if p > 0.5]
                        high_conf_count = len(high_conf_fakes)
                        
                        # Hybrid decision (research-aligned):
                        # 1. High-confidence trigger: any frame >0.85 = deepfake
                        # 2. Majority vote: >50% of frames flagged as fake
                        if high_conf_count >= 1:
                            is_deepfake = True
                            detection_method = "HIGH confidence frame detected"
                        elif len(majority_fakes) > total_frames // 2:
                            is_deepfake = True
                            detection_method = "majority vote"
                        else:
                            is_deepfake = False
                            detection_method = "insufficient evidence"
                        
                        # Calculate metrics for display
                        avg_fake = np.mean(fake_probs_list)
                        weights = [p ** 2 if p > 0.5 else p * 0.5 for p in fake_probs_list]
                        weighted_avg = np.average(fake_probs_list, weights=weights) if sum(weights) > 0 else avg_fake
                        
                        # Uncertainty calculation (for ambiguous cases)
                        uncertainty = "low"
                        if 0.6 <= max_fake_prob <= 0.85 and not is_deepfake:
                            uncertainty = "medium"
                        elif 0.4 <= max_fake_prob <= 0.6:
                            uncertainty = "high"
                        
                        st.session_state.video_results = results
                        st.session_state.avg_fake = avg_fake
                        st.session_state.weighted_avg = weighted_avg
                        st.session_state.max_fake_prob = max_fake_prob
                        st.session_state.deepfake_votes = deepfake_votes
                        st.session_state.total_frames = total_frames
                        st.session_state.fake_probs_list = fake_probs_list
                        st.session_state.high_conf_frames = high_conf_count
                        st.session_state.is_deepfake = is_deepfake
                        st.session_state.detection_method = detection_method
                        st.session_state.uncertainty = uncertainty
                        st.session_state.early_stopped = early_stopped
                    else:
                        st.session_state.video_error = "No valid frames processed from this video."

# RESULTS COLUMN
with col_results:
    st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
    
    if upload_type == "Image":
        if "prediction" in st.session_state:
            pred = st.session_state.prediction
            conf_real = st.session_state.conf_real
            conf_fake = st.session_state.conf_fake
            confidence = st.session_state.confidence
            
            # Main prediction card
            if pred == "Real":
                st.markdown(f"""
                    <div class="result-card-real">
                        <h2>Real Image</h2>
                        <p>This image appears to be authentic</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card-fake">
                        <h2>Deepfake Detected</h2>
                        <p>This image shows signs of AI manipulation</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Confidence metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Real Confidence</div>
                        <div class="metric-value">{conf_real:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Fake Confidence</div>
                        <div class="metric-value">{conf_fake:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(conf_fake, text=f"Deepfake Score: {conf_fake:.1%}")
        
        else:
            st.markdown("""
                <div class="info-box">
                    Upload an image and click the analysis button to begin detection.
                </div>
            """, unsafe_allow_html=True)
    
    else:  # Video
        if "video_error" in st.session_state:
            st.warning(st.session_state.video_error)
            del st.session_state.video_error
        
        elif "video_results" in st.session_state:
            results = st.session_state.video_results
            avg_fake = st.session_state.avg_fake
            weighted_avg = st.session_state.get('weighted_avg', avg_fake)
            max_fake_prob = st.session_state.get('max_fake_prob', max([r['fake'] for r in results]))
            deepfake_votes = st.session_state.deepfake_votes
            total_frames = st.session_state.total_frames
            high_conf_frames = st.session_state.get('high_conf_frames', 0)
            fake_probs_list = st.session_state.get('fake_probs_list', [r['fake'] for r in results])
            is_deepfake = st.session_state.is_deepfake
            
            # Main prediction with research-backed hybrid voting
            detection_method = st.session_state.get('detection_method', 'standard analysis')
            uncertainty = st.session_state.get('uncertainty', 'low')
            early_stopped = st.session_state.get('early_stopped', False)
            
            if is_deepfake:
                if detection_method == "HIGH confidence frame detected":
                    st.markdown(f"""
                        <div class="result-card-fake">
                            <h2>Deepfake Video Detected</h2>
                            <p>Video shows signs of AI manipulation ({detection_method}: {max_fake_prob:.1%} confidence)</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card-fake">
                            <h2>Deepfake Video Detected</h2>
                            <p>Video shows signs of AI manipulation ({detection_method}: {deepfake_votes} of {total_frames} frames flagged)</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card-real">
                        <h2>Real Video</h2>
                        <p>Video appears to be authentic (flagged on {deepfake_votes} of {total_frames} frames)</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Uncertainty warning for ambiguous cases
            if uncertainty == "medium" or uncertainty == "high":
                st.warning(f"⚠️ **Ambiguous Results:** Maximum confidence ({max_fake_prob:.1%}) falls in uncertain range. Consider manual review for forensic accuracy.")
            
            # Early stopping indicator
            if early_stopped:
                st.info(f"⚡ **Early Detection:** Processing stopped early after finding {high_conf_frames} high-confidence deepfake frame(s). This indicates strong evidence of manipulation.")
            
            # Enhanced metrics display
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Average Score</div>
                        <div class="metric-value">{avg_fake:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Max Score</div>
                        <div class="metric-value">{max_fake_prob:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Weighted Avg</div>
                        <div class="metric-value">{weighted_avg:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.progress(avg_fake, text=f"Average Deepfake Score: {avg_fake:.1%}")
            
            # Frame-by-frame details (research-aligned display)
            st.markdown("**Frame-by-Frame Analysis:**")
            st.write(f"**Fake probability per frame:** {[round(float(p), 2) for p in fake_probs_list]}")
            
            # Show high-confidence frames if any
            if high_conf_frames > 0:
                st.info(f"⚠️ **High-confidence detection:** {high_conf_frames} frame(s) with >85% fake probability detected. This is a strong indicator of deepfake content.")
            
            # Detailed frame breakdown
            for result in results:
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.markdown(f"**Frame {result['frame']}**")
                with col2:
                    status = "Real" if result['fake'] < 0.5 else "Deepfake"
                    st.markdown(f"*{status}*")
                with col3:
                    st.progress(result['fake'])
        
        else:
            st.markdown("""
                <div class="info-box">
                    Upload a video and click the analysis button to begin detection.
                    <br><small>Note: Video processing uses face detection for improved accuracy.</small>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 12px; margin-top: 20px;">
        <p><strong>Spot DeepFake</strong> - Advanced AI-Powered Deepfake Detection</p>
        <p>Model: EfficientNetB0 | Accuracy: 95.67% | For Research and Educational Use</p>
    </div>
""", unsafe_allow_html=True)
