import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_manager import DatasetManager
from trainers.logistic_trainer import LogisticTrainer
from trainers.random_forest_trainer import RandomForestTrainer
from trainers.cnn_trainer import CNN
from inference.predictor import Predictor
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2

# Page config
st.set_page_config(
    page_title="Teachable Machine Clone",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dataset_manager' not in st.session_state:
    st.session_state.dataset_manager = DatasetManager()
if 'predictor' not in st.session_state:
    st.session_state.predictor = Predictor()
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'uploaded_files_cache' not in st.session_state:
    st.session_state.uploaded_files_cache = []

# Header
st.markdown('<div class="main-header">🤖 Teachable Machine Clone</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Train ML models with your images in real-time!</p>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Collection", "🎓 Train Models", "🔮 Predictions", "📈 Model Comparison"])

# ===== TAB 1: DATA COLLECTION =====
with tab1:
    st.markdown('<div class="sub-header">📊 Collect Training Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Images")
        
        # Initialize form reset counter
        if 'form_counter' not in st.session_state:
            st.session_state.form_counter = 0
        
        # Class name input with unique key based on counter
        class_name = st.text_input("Class Name", placeholder="e.g., Cat, Dog, Bird", key=f"class_name_{st.session_state.form_counter}")
        
        # File uploader with unique key based on counter
        uploaded_files = st.file_uploader(
            "Choose images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload at least 5 images per class",
            key=f"image_uploader_{st.session_state.form_counter}"
        )
        
        # Show number of uploaded files
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} image(s) selected")
        
        # Buttons in columns
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("Add Images to Class", type="primary", disabled=not class_name or not uploaded_files, use_container_width=True):
                with st.spinner("Processing images..."):
                    # Reset file pointers before saving
                    for f in uploaded_files:
                        f.seek(0)
                    saved_count, errors = st.session_state.dataset_manager.save_images(class_name, uploaded_files)
                    
                    if saved_count > 0:
                        st.success(f"✅ Successfully added {saved_count} images to class '{class_name}'")
                        # Increment counter to reset form
                        st.session_state.form_counter += 1
                        st.rerun()
                    
                    if errors:
                        with st.expander("⚠️ Errors encountered"):
                            for error in errors:
                                st.warning(error)
        
        with btn_col2:
            if st.button("🗑️ Clear Form", type="secondary", disabled=not (class_name or uploaded_files), use_container_width=True):
                # Increment counter to reset form
                st.session_state.form_counter += 1
                st.rerun()
    
    with col2:
        st.subheader("Dataset Overview")
        
        class_info = st.session_state.dataset_manager.get_class_info()
        stats = st.session_state.dataset_manager.get_dataset_stats()
        
        # Display metrics
        st.metric("Total Classes", stats['num_classes'])
        st.metric("Total Images", stats['total_images'])
        
        ready, message = st.session_state.dataset_manager.is_ready_for_training()
        if ready:
            st.success(f"✅ {message}")
        else:
            st.warning(f"⚠️ {message}")
        
        # Display class distribution
        if class_info:
            st.markdown("**Class Distribution:**")
            df = pd.DataFrame(list(class_info.items()), columns=['Class', 'Images'])
            
            fig = px.bar(df, x='Class', y='Images', 
                        title='Images per Class',
                        color='Images',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True, key='class_distribution_chart')
    
    # Class management
    if class_info:
        st.markdown('<div class="sub-header">Manage Classes</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(class_info))
        for idx, (cls_name, count) in enumerate(class_info.items()):
            with cols[idx]:
                st.info(f"**{cls_name}**\n\n{count} images")
                if st.button(f"Delete {cls_name}", key=f"del_{cls_name}"):
                    if st.session_state.dataset_manager.delete_class(cls_name):
                        st.success(f"Deleted class '{cls_name}'")
                        st.rerun()
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
                if st.session_state.dataset_manager.clear_all_data():
                    st.success("All data cleared!")
                    st.rerun()
        
        with col_btn2:
            if st.button("🔄 Reset Everything", type="secondary", use_container_width=True):
                if st.session_state.dataset_manager.reset_all_data():
                    st.session_state.trained_models = {}
                    st.session_state.predictor = Predictor()
                    st.success("✅ Reset complete! All data and models cleared.")
                    st.rerun()

# ===== TAB 2: TRAIN MODELS =====
with tab2:
    st.markdown('<div class="sub-header">🎓 Train Machine Learning Models</div>', unsafe_allow_html=True)
    
    ready, message = st.session_state.dataset_manager.is_ready_for_training()
    
    if not ready:
        st.warning(f"⚠️ {message}")
        st.info("Please add more images in the Data Collection tab before training.")
    else:
        st.success(f"✅ Dataset ready for training!")
        
        # Model selection
        st.subheader("Select Models to Train")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_lr = st.checkbox("Logistic Regression", value=True)
        with col2:
            train_rf = st.checkbox("Random Forest", value=True)
        with col3:
            train_cnn = st.checkbox("CNN (TensorFlow/Keras)", value=True)
        
        # CNN epochs
        if train_cnn:
            epochs = st.slider("CNN Training Epochs", min_value=5, max_value=50, value=20, step=5)
        
        # Train button
        if st.button("🚀 Start Training", type="primary", disabled=not (train_lr or train_rf or train_cnn)):
            # Load dataset
            with st.spinner("Loading dataset..."):
                X, y, class_names = st.session_state.dataset_manager.load_dataset()
                st.info(f"Loaded {len(X)} images from {len(class_names)} classes")
            
            results = {}
            
            # Train Logistic Regression
            if train_lr:
                st.markdown("### Training Logistic Regression")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    status_text.text(message)
                    progress_bar.progress(progress / 100)
                
                trainer = LogisticTrainer()
                result = trainer.train(X, y, class_names, progress_callback=update_progress)
                results['Logistic Regression'] = result
                st.success(f"✅ Training complete! Test Accuracy: {result['test_accuracy']:.3f}")
            
            # Train Random Forest
            if train_rf:
                st.markdown("### Training Random Forest")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    status_text.text(message)
                    progress_bar.progress(progress / 100)
                
                trainer = RandomForestTrainer()
                result = trainer.train(X, y, class_names, progress_callback=update_progress)
                results['Random Forest'] = result
                st.success(f"✅ Training complete! Test Accuracy: {result['test_accuracy']:.3f}")
            
            # Train CNN
            if train_cnn:
                st.markdown("### Training CNN (TensorFlow/Keras)")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    status_text.text(message)
                    progress_bar.progress(progress / 100)
                
                trainer = CNN()
                result = trainer.train(X, y, class_names, epochs=epochs, progress_callback=update_progress)
                results['CNN (TensorFlow/Keras)'] = result
                st.success(f"✅ Training complete! Test Accuracy: {result['test_accuracy']:.3f}")
            
            # Store results
            st.session_state.trained_models = results
            
            # Display results
            st.markdown("### Training Results")
            
            for model_name, result in results.items():
                # Model icon mapping
                model_icons = {
                    'Logistic Regression': '📊',
                    'Random Forest': '🌳',
                    'CNN (TensorFlow/Keras)': '🧠'
                }
                icon = model_icons.get(model_name, '🤖')
                
                with st.expander(f"{icon} {model_name} Results", expanded=True):
                    # Metrics with better styling
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Train Accuracy", f"{result['train_accuracy']:.1%}", 
                                 help="Model performance on training data")
                    with col2:
                        st.metric("✅ Test Accuracy", f"{result['test_accuracy']:.1%}", 
                                 delta=f"{(result['test_accuracy']-result['train_accuracy']):.1%}",
                                 help="Model performance on unseen test data")
                    with col3:
                        st.metric("⏱️ Training Time", f"{result['training_time']:.2f}s",
                                 help="Total time taken to train the model")
                    
                    # Confusion matrix
                    st.markdown("**Confusion Matrix:**")
                    cm = result['confusion_matrix']
                    fig = px.imshow(cm, 
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=class_names, 
                                   y=class_names,
                                   text_auto=True,
                                   color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True, key=f'confusion_matrix_{model_name.replace(" ", "_")}')
                    
                    # Classification report
                    st.markdown("**Classification Report:**")
                    report_df = pd.DataFrame(result['classification_report']).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# ===== TAB 3: PREDICTIONS =====
with tab3:
    st.markdown('<div class="sub-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    # Load models
    if st.button("🔄 Load Trained Models"):
        with st.spinner("Loading models..."):
            if st.session_state.predictor.load_all_models():
                loaded_models = st.session_state.predictor.get_available_models()
                st.success(f"✅ Loaded {len(loaded_models)} model(s)")
                st.info(f"**Available models:** {', '.join(loaded_models)}")
            else:
                st.error("❌ No trained models found. Please train models first.")
    
    if st.session_state.predictor.is_ready():
        st.markdown("---")
        prediction_mode = st.radio("🎯 Select Prediction Mode", ["📤 Upload Image", "📹 Webcam (Live)"], horizontal=True)
        
        if prediction_mode == "📤 Upload Image":
            st.subheader("📤 Upload Image for Prediction")
            
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'bmp'])
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    st.markdown("**Predictions from All Models:**")
                    
                    results = st.session_state.predictor.predict_all_models(image)
                    
                    for model_name, (predicted_class, probabilities) in results.items():
                        with st.expander(f"🤖 {model_name}", expanded=True):
                            st.success(f"**Predicted Class:** {predicted_class}")
                            
                            # Probability bar chart
                            prob_df = pd.DataFrame(list(probabilities.items()), 
                                                  columns=['Class', 'Probability'])
                            prob_df = prob_df.sort_values('Probability', ascending=False)
                            
                            fig = px.bar(prob_df, x='Class', y='Probability',
                                       color='Probability',
                                       color_continuous_scale='viridis',
                                       range_y=[0, 1])
                            st.plotly_chart(fig, use_container_width=True, key=f'prediction_prob_{model_name.replace(" ", "_")}')
        
        else:  # Webcam mode
            st.subheader("📹 Live Webcam Predictions")
            st.info("📷 Click **Capture from Webcam** to take a snapshot and get predictions.")
            
            # Initialize webcam state
            if 'webcam_running' not in st.session_state:
                st.session_state.webcam_running = False
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("📸 Capture from Webcam", type="primary", use_container_width=True):
                    with st.spinner("📷 Capturing..."):
                        try:
                            # Initialize webcam with optimized settings
                            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend on Windows for speed
                            
                            if not cap.isOpened():
                                st.error("❌ Could not access webcam. Please check your camera permissions.")
                            else:
                                # Set lower resolution for speed
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                
                                # Capture frame immediately
                                ret, frame = cap.read()
                                cap.release()
                                
                                if ret:
                                    # Convert BGR to RGB
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    image = Image.fromarray(frame_rgb)
                                    
                                    # Store in session state
                                    st.session_state.webcam_image = image
                                    st.success("✅ Image captured!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to capture image from webcam.")
                        except Exception as e:
                            st.error(f"❌ Error accessing webcam: {str(e)}")
            
            with col_btn2:
                if st.button("🗑️ Clear Capture", type="secondary", use_container_width=True):
                    if 'webcam_image' in st.session_state:
                        del st.session_state.webcam_image
                        st.rerun()
            
            # Display captured image and predictions
            if 'webcam_image' in st.session_state:
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(st.session_state.webcam_image, caption="Captured Image", use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Predictions")
                    
                    # Cache predictions to avoid recomputing
                    if 'last_prediction_results' not in st.session_state or st.session_state.get('last_image_id') != id(st.session_state.webcam_image):
                        with st.spinner("Analyzing..."):
                            results = st.session_state.predictor.predict_all_models(st.session_state.webcam_image)
                            st.session_state.last_prediction_results = results
                            st.session_state.last_image_id = id(st.session_state.webcam_image)
                    else:
                        results = st.session_state.last_prediction_results
                    
                    for model_name, (predicted_class, probabilities) in results.items():
                        # Get top prediction confidence
                        top_confidence = max(probabilities.values())
                        
                        with st.expander(f"🤖 {model_name}", expanded=True):
                            st.success(f"**{predicted_class}** ({top_confidence:.1%})")
                            
                            # Display top 3 probabilities only
                            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
                            for cls, prob in sorted_probs:
                                st.progress(prob, text=f"{cls}: {prob:.1%}")
    else:
        st.warning("⚠️ No models loaded. Click 'Load Trained Models' button above.")

# ===== TAB 4: MODEL COMPARISON =====
with tab4:
    st.markdown('<div class="sub-header">📈 Model Comparison</div>', unsafe_allow_html=True)
    
    if st.session_state.trained_models:
        models_data = []
        for model_name, result in st.session_state.trained_models.items():
            models_data.append({
                'Model': model_name,
                'Train Accuracy': result['train_accuracy'],
                'Test Accuracy': result['test_accuracy'],
                'Training Time (s)': result['training_time']
            })
        
        df = pd.DataFrame(models_data)
        
        # Comparison table
        st.markdown("### Performance Comparison")
        st.dataframe(df.style.format({
            'Train Accuracy': '{:.3f}',
            'Test Accuracy': '{:.3f}',
            'Training Time (s)': '{:.2f}'
        }).highlight_max(subset=['Test Accuracy'], color='lightgreen'), use_container_width=True)
        
        # Accuracy comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Train Accuracy', x=df['Model'], y=df['Train Accuracy']))
            fig.add_trace(go.Bar(name='Test Accuracy', x=df['Model'], y=df['Test Accuracy']))
            fig.update_layout(title='Accuracy Comparison', barmode='group', yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True, key='accuracy_comparison_chart')
        
        with col2:
            fig = px.bar(df, x='Model', y='Training Time (s)',
                        title='Training Time Comparison',
                        color='Training Time (s)',
                        color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True, key='training_time_comparison_chart')
        
        # CNN training history
        if 'CNN (TensorFlow/Keras)' in st.session_state.trained_models:
            st.markdown("### CNN Training History")
            cnn_result = st.session_state.trained_models['CNN (TensorFlow/Keras)']
            history = cnn_result['history']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Train Accuracy'))
                fig.add_trace(go.Scatter(y=history['val_accuracy'], mode='lines', name='Val Accuracy'))
                fig.update_layout(title='Accuracy over Epochs', xaxis_title='Epoch', yaxis_title='Accuracy')
                st.plotly_chart(fig, use_container_width=True, key='cnn_accuracy_history_chart')
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Train Loss'))
                fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Val Loss'))
                fig.update_layout(title='Loss over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig, use_container_width=True, key='cnn_loss_history_chart')
    else:
        st.info("📊 No training results available. Train models first!")