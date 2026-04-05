import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import InputLayer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Custom InputLayer to handle extra kwargs from newer Keras
class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Remove only the 'optional' argument – keep batch_shape
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

custom_objects = {'InputLayer': CustomInputLayer}

# -------------------------------------------------------------------
# Page configuration
st.set_page_config(page_title="Breast Cancer Detection AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; color: white; }
    .prediction-healthy { background-color: #28a745; color: white; padding: 1rem; border-radius: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .prediction-sick { background-color: #dc3545; color: white; padding: 1rem; border-radius: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .confidence-meter { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
    .info-box { background-color: #e3f2fd; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196f3; margin: 1rem 0; }
    .info-box-dark {
        background-color: #2d2d2d;
        color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #bb86fc;
        margin: 1rem 0;
    }

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Score-CAM (as used in training)
def score_cam(model, img_array, last_conv_layer_name='block5_conv3', class_index=0):
    conv_layer = model.get_layer(last_conv_layer_name)
    conv_model = Model(inputs=model.input, outputs=conv_layer.output)
    conv_outputs = conv_model.predict(img_array, verbose=0)[0]

    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i in range(conv_outputs.shape[-1]):
        act = conv_outputs[:, :, i]
        act -= act.min()
        if act.max() != 0:
            act /= act.max()
        act_resized = cv2.resize(act, (img_array.shape[2], img_array.shape[1]))
        masked_input = img_array.copy()
        masked_input[0] *= act_resized[..., np.newaxis]
        score = model.predict(masked_input, verbose=0)[0][class_index]
        cam += score * act

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam
# -------------------------------------------------------------------
# Load models (cached)
@st.cache_resource
def load_models():
    model_path = 'models/'
    if not os.path.exists(model_path):
        st.error(f"❌ Models folder not found at '{model_path}'")
        return None

    # We'll use the .keras files (the original saved ones)
    required = [
        'feature_extractor.keras',
        'breast_cancer_model.keras',
        'ridge_clf.pkl',
        'lda_clf.pkl',
        'extra_clf.pkl',
        'lgbm_clf.pkl',
        'cs_selector.pkl',
        'ga_selected_idx.npy'
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        return None

    try:
        with st.spinner("Loading models..."):
            feature_extractor = load_model(
                os.path.join(model_path, 'feature_extractor.keras'),
                custom_objects=custom_objects,
                compile=False
            )
            full_model = load_model(
                os.path.join(model_path, 'breast_cancer_model.keras'),
                custom_objects=custom_objects,
                compile=False
            )

            ridge = joblib.load(os.path.join(model_path, 'ridge_clf.pkl'))
            lda = joblib.load(os.path.join(model_path, 'lda_clf.pkl'))
            extra = joblib.load(os.path.join(model_path, 'extra_clf.pkl'))
            lgbm = joblib.load(os.path.join(model_path, 'lgbm_clf.pkl'))

            cs_selector = joblib.load(os.path.join(model_path, 'cs_selector.pkl'))
            ga_idx = np.load(os.path.join(model_path, 'ga_selected_idx.npy'))

            threshold = 0.5
            if os.path.exists(os.path.join(model_path, 'best_threshold.npy')):
                threshold = np.load(os.path.join(model_path, 'best_threshold.npy')).item()

        # Optional debug info (can be removed)
        #st.sidebar.write(f"ridge type: {type(ridge)}")
        #st.sidebar.write(f"lda type: {type(lda)}")
        #st.sidebar.write(f"extra type: {type(extra)}")
        #st.sidebar.write(f"lgbm type: {type(lgbm)}")

        st.success("✅ All models loaded successfully!")
        return {
            'feature_extractor': feature_extractor,
            'full_model': full_model,
            'ridge': ridge, 'lda': lda, 'extra': extra, 'lgbm': lgbm,
            'cs_selector': cs_selector, 'ga_idx': ga_idx, 'threshold': threshold
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# -------------------------------------------------------------------
# Prediction function (with fallback for Ridge)
def predict_image(image, models):
    # Preprocess the image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_array = np.expand_dims(img.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    # 1. Extract features from the CNN
    features = models['feature_extractor'].predict(img_array, verbose=0)
    #st.write(f"Features shape: {features.shape}, min={features.min():.4f}, max={features.max():.4f}")

    # 2. Apply GA‑selected indices
    features_ga = features[:, models['ga_idx']]
    #st.write(f"After GA (selected indices) shape: {features_ga.shape}, min={features_ga.min():.4f}, max={features_ga.max():.4f}")

    # 3. Apply CS (cuckoo search) selector
    features_opt = models['cs_selector'].transform(features_ga)
    #st.write(f"After CS (final) shape: {features_opt.shape}, min={features_opt.min():.4f}, max={features_opt.max():.4f}")

    # 4. Predict probabilities with the ensemble classifiers
    try:
        prob_ridge = models['ridge'].predict_proba(features_opt)[0, 1]
    except Exception as e:
        st.warning(f"Ridge error: {e}. Using fallback probability 0.5.")
        prob_ridge = 0.5

    prob_lda = models['lda'].predict_proba(features_opt)[0, 1]
    prob_extra = models['extra'].predict_proba(features_opt)[0, 1]
    prob_lgbm = models['lgbm'].predict_proba(features_opt)[0, 1]

    #st.write(f"Probabilities: Ridge={prob_ridge:.4f}, LDA={prob_lda:.4f}, Extra={prob_extra:.4f}, LGBM={prob_lgbm:.4f}")

    # 5. Ensemble average and final prediction
    ensemble_prob = (prob_ridge + prob_lda + prob_extra + prob_lgbm) / 4
    prediction = int(ensemble_prob >= models['threshold'])
    confidence_in_pred = ensemble_prob if prediction == 1 else 1 - ensemble_prob
    
    # 6. Score-CAM heatmap (with the full model)
    heatmap = score_cam(models['full_model'], img_array)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return {
    'prediction': prediction,
    'confidence': confidence_in_pred,          # <-- changed
    'prob_abnormal': ensemble_prob,            # <-- keep raw probability (optional)
    'probabilities': {
        'Ridge': prob_ridge,
        'LDA': prob_lda,
        'Extra Trees': prob_extra,
        'LightGBM': prob_lgbm,
        'Ensemble': ensemble_prob
    },
    'original_img': img,
    'overlay_img': overlay_rgb,
    'heatmap': heatmap
}
   
# -------------------------------------------------------------------
# UI
st.markdown("""
<div class="main-header">
    <h1>🩺Breast Cancer Detection</h1>
    <p>Advanced thermal image analysis using Deep Learning + Ensemble Methods</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://media.istockphoto.com/id/1496508204/vector/abstract-vector-heat-map-of-hot-and-cold-distribution-background.jpg?s=612x612&w=0&k=20&c=d3u0hzoTOBuK9qxhyaNtQyX7nrNa92aAm_4WGxOOzVA=", width=80)
    st.title("About")
    st.markdown("""
    ### 🩺 About This App
    Uses VGG16 + GA + CS + Ensemble of 4 classifiers.
    """)
    st.divider()
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", "97.1%")
    st.metric("AUC-ROC", "0.99")
    st.metric("Sensitivity", "96.7%")
    st.metric("Specificity", "97.25%")

models = load_models()
if models is None:
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 📤 Upload Thermal Image")
    uploaded_file = st.file_uploader("Choose a thermal breast image...", type=['jpg', 'jpeg', 'png', 'bmp'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file:
        st.markdown("### 🔬 Analysis Results")
        with st.spinner("🧠 Analyzing image..."):
            result = predict_image(image, models)

        if result:
            if result['prediction'] == 1:
                st.markdown('<div class="prediction-sick">⚠️ DIAGNOSIS: SICK (Abnormal)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-healthy">✅ DIAGNOSIS: HEALTHY (Normal)</div>', unsafe_allow_html=True)

            conf = result['confidence'] * 100
            st.markdown('<div class="confidence-meter">', unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {conf:.1f}%")
            st.progress(conf / 100)
            st.markdown('</div>', unsafe_allow_html=True)

            # Show numerical confidence values (probability of predicted class)
            st.markdown("#### 📊 Model Confidence (Probability of the Predicted Class)")

            # For each model, compute confidence in the overall prediction (ensemble's decision)
            overall_pred = result['prediction']
            conf_data = []
            for name, prob_abnormal in result['probabilities'].items():
                conf = prob_abnormal if overall_pred == 1 else 1 - prob_abnormal
                conf_data.append({"Model": name, "Confidence (%)": f"{conf*100:.1f}%"})

            # Display as a simple table without graphs
            st.dataframe(pd.DataFrame(conf_data), use_container_width=True, hide_index=True)

            # Optional: show the raw abnormality probabilities too (toggle with expander)
            with st.expander("📈 Show raw probability of abnormality"):
                raw_df = pd.DataFrame({
                    "Model": list(result['probabilities'].keys()),
                    "Abnormality Probability (%)": [f"{p*100:.1f}%" for p in result['probabilities'].values()]
                })
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
if uploaded_file and result:
    st.divider()
    st.markdown("### 📊 Visualizations & Explainability")
    tab1, tab2, tab3 = st.tabs(["🎨 Heatmap", "📈 Confidence Analysis", "ℹ️ Interpretation"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(result['original_img'], caption="Original Image", use_container_width=True)
        with col_b:
            st.image(result['overlay_img'], caption="AI Attention Heatmap", use_container_width=True)
        

    with tab2:
        fig = go.Figure()
        models_names = list(result['probabilities'].keys())
        confidences = [p * 100 for p in result['probabilities'].values()]
        colors = ['#28a745' if c < 50 else '#dc3545' for c in confidences]
        fig.add_trace(go.Bar(x=models_names, y=confidences, marker_color=colors,
                             text=[f'{c:.1f}%' for c in confidences], textposition='auto'))
        fig.add_hline(y=models['threshold']*100, line_dash="dash", line_color="gray",
                      annotation_text=f"Threshold ({models['threshold']*100:.0f}%)", annotation_position="bottom right")
        fig.update_layout(title="Model Confidence Comparison", yaxis_range=[0, 100], height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if result['prediction'] == 1:
            st.markdown(f"""
            <div class="info-box-dark">
                <b>⚠️ Interpretation: Abnormal Result</b><br><br>
                Ensemble confidence: {result['confidence']*100:.1f}%<br>
                <b>Next steps:</b> Consult a healthcare professional.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box-dark">
                <b>✅ Interpretation: Normal Result</b><br><br>
                Ensemble confidence: {result['confidence']*100:.1f}%<br>
                <b>Important:</b> Regular screening is still recommended.
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. Not for clinical diagnosis.</p>
</div>
""", unsafe_allow_html=True)