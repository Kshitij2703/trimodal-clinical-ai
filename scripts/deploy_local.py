"""
LOCAL DEPLOYMENT - Gradio Demo for Tri-Modal Clinical AI
Loads trained models from Kaggle and runs inference locally
Updated for EfficientNet-B0 backbone
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
CONFIG = {
    'batch_size': 16,
    'num_classes': 8,
    'vision_dim': 1280,              # EfficientNet-B0 output
    'text_dim': 768,
    'metadata_dim': 2,
    'vision_embed_dim': 256,
    'text_embed_dim': 256,
    'metadata_embed_dim': 16,
    'bottleneck_dim': 256,
    'hdc_dim': 4096,
    'dropout_vision': 0.5,
    'dropout_text': 0.4,
    'dropout_fusion': 0.5,
}

LABEL_COLUMNS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                 'Effusion', 'Emphysema', 'Pneumonia', 'Pneumothorax']

# ===== PATHS =====
MODEL_DIR = Path(__file__).parent  # models are in the same directory as this script
MODEL_PATH = MODEL_DIR / "efficientnet_fused.pth"
HDC_INDEX_PATH = MODEL_DIR / "retrieval_index.pkl"

# ===== MODEL ARCHITECTURE (same as training) =====
class TriModalClinicalAI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # EfficientNet-B0 vision encoder
        efficientnet = models.efficientnet_b0(weights=None)
        self.vision_encoder = efficientnet.features
        self.vision_pool = nn.AdaptiveAvgPool2d(1)
        
        self.vision_proj = nn.Sequential(
            nn.Linear(config['vision_dim'], config['vision_embed_dim']),
            nn.BatchNorm1d(config['vision_embed_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_vision'])
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(config['text_dim'], config['text_embed_dim']),
            nn.BatchNorm1d(config['text_embed_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_text'])
        )
        
        self.metadata_proj = nn.Sequential(
            nn.Linear(config['metadata_dim'], config['metadata_embed_dim']),
            nn.ReLU()
        )
        
        fusion_input_dim = (config['vision_embed_dim'] + 
                           config['text_embed_dim'] + 
                           config['metadata_embed_dim'])
        
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_input_dim, config['bottleneck_dim']),
            nn.BatchNorm1d(config['bottleneck_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_fusion'])
        )
        
        self.classifier = nn.Linear(config['bottleneck_dim'], config['num_classes'])
        self.temperature = nn.Parameter(torch.ones(1))
    
    def _encode_vision(self, x):
        features = self.vision_encoder(x)
        pooled   = self.vision_pool(features)
        return pooled.flatten(1)

    def forward(self, frontal, lateral, text, metadata, 
                return_bottleneck=False, use_temperature=False):
        frontal_feat   = self._encode_vision(frontal)
        lateral_feat   = self._encode_vision(lateral)
        vision_feat    = (frontal_feat + lateral_feat) / 2.0
        vision_embed   = self.vision_proj(vision_feat)
        
        text_embed     = self.text_proj(text)
        metadata_embed = self.metadata_proj(metadata)
        
        fused      = torch.cat([vision_embed, text_embed, metadata_embed], dim=1)
        bottleneck = self.fusion_bottleneck(fused)
        logits     = self.classifier(bottleneck)
        
        if use_temperature:
            logits = logits / self.temperature
        
        if return_bottleneck:
            return logits, bottleneck
        return logits

# ===== GLOBAL STATE =====
model = None
hdc_index = None
bert_model = None
bert_tokenizer = None
device = None

def load_models():
    """Load all models and indices"""
    global model, hdc_index, bert_model, bert_tokenizer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if models exist
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            f"Please download trained models from Kaggle and place in {MODEL_DIR}/"
        )
    
    if not HDC_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"HDC index not found at {HDC_INDEX_PATH}\n"
            f"Please download retrieval_index.pkl from Kaggle and place in {MODEL_DIR}/"
        )
    
    # Load classifier
    print("Loading tri-modal classifier (EfficientNet-B0)...")
    model = TriModalClinicalAI(CONFIG).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✓ Classifier loaded")
    
    # Load HDC index
    print("Loading HDC retrieval index...")
    with open(HDC_INDEX_PATH, 'rb') as f:
        hdc_index = pickle.load(f)
    print(f"✓ HDC index loaded ({len(hdc_index['hypervectors'])} training cases)")
    
    # Load BERT
    print("Loading ClinicalBERT...")
    bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_attentions=True)
    bert_model.eval()
    print("✓ ClinicalBERT loaded")
    
    print("\n✓ All models ready for inference!")

def preprocess_image(image):
    """Preprocess image for model"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def extract_bert_embedding(text):
    """Extract 768-dim BERT embedding"""
    inputs = bert_tokenizer(text, max_length=256, padding='max_length', 
                           truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
    
    return embedding

def retrieve_similar(bottleneck_embedding):
    """Retrieve top-3 similar cases using HDC"""
    query_hv = np.sign(bottleneck_embedding @ hdc_index['projection_matrix'])
    query_hv[query_hv == 0] = 1
    query_hv = query_hv.astype(np.int8)
    
    train_hvs = hdc_index['hypervectors']
    distances = np.sum(train_hvs != query_hv, axis=1)
    top_3_idx = np.argsort(distances)[:3]
    
    results = []
    for idx in top_3_idx:
        similarity = 1.0 - (distances[idx] / len(query_hv))
        labels = hdc_index['train_labels'][idx]
        label_str = ', '.join([LABEL_COLUMNS[i] for i in range(len(labels)) if labels[i] == 1])
        
        results.append({
            'uid': hdc_index['train_uids'][idx],
            'similarity': f"{similarity*100:.1f}%",
            'labels': label_str if label_str else 'Normal'
        })
    
    return results

def predict(frontal_image, lateral_image, report_text):
    """Main prediction function"""
    
    if frontal_image is None:
        return "⚠️ Please upload a frontal X-ray image", "", "", ""
    
    # Use frontal as lateral fallback
    if lateral_image is None:
        lateral_image = frontal_image
    
    # Default report
    if not report_text or report_text.strip() == "":
        report_text = "No clinical findings reported."
    
    try:
        # Preprocess
        frontal_tensor = preprocess_image(frontal_image)
        lateral_tensor = preprocess_image(lateral_image)
        text_embedding = extract_bert_embedding(report_text)
        text_tensor = torch.FloatTensor(text_embedding)
        metadata_tensor = torch.FloatTensor([1.0, 0.0])
        
        # Inference
        with torch.no_grad():
            frontal_batch = frontal_tensor.unsqueeze(0).to(device)
            lateral_batch = lateral_tensor.unsqueeze(0).to(device)
            text_batch = text_tensor.unsqueeze(0).to(device)
            metadata_batch = metadata_tensor.unsqueeze(0).to(device)
            
            logits, bottleneck = model(frontal_batch, lateral_batch, text_batch, 
                                      metadata_batch, return_bottleneck=True)
            
            predictions = torch.sigmoid(logits)[0].cpu().numpy()
            bottleneck_np = bottleneck[0].cpu().numpy()
        
        # Format predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        diagnosis_text = "## 🎯 Predicted Diagnoses\n\n"
        for idx in top_indices:
            prob = predictions[idx]
            if prob > 0.1:  # Show predictions > 10%
                diagnosis_text += f"- **{LABEL_COLUMNS[idx]}**: {prob:.1%}\n"
        
        # Retrieve similar cases
        similar_cases = retrieve_similar(bottleneck_np)
        retrieval_text = "## 🔍 Similar Confirmed Cases (HDC Retrieval)\n\n"
        for i, case in enumerate(similar_cases):
            retrieval_text += f"**Case {i+1}** - Patient {case['uid']}\n"
            retrieval_text += f"- Similarity: {case['similarity']}\n"
            retrieval_text += f"- Diagnoses: {case['labels']}\n\n"
        
        # BERT attention
        inputs = bert_tokenizer(report_text, return_tensors='pt', max_length=256, truncation=True)
        tokens = bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            attentions = outputs.attentions[-1]
            cls_attention = attentions[0, :, 0, :].mean(dim=0).numpy()
        
        top_token_idx = np.argsort(cls_attention)[-10:][::-1]
        attention_text = "## 📊 Key Clinical Terms (BERT Attention)\n\n"
        for idx in top_token_idx:
            if tokens[idx] not in ['[CLS]', '[SEP]', '[PAD]']:
                attention_text += f"- **{tokens[idx]}** ({cls_attention[idx]:.3f})\n"
        
        # Metadata
        metadata_text = (
            f"## ℹ️ Metadata\n\n"
            f"- View Position: Frontal {'+ Lateral' if lateral_image is not None else '(only)'}\n"
            f"- Model: Tri-Modal EfficientNet-B0 + ClinicalBERT\n"
            f"- Device: {device}\n"
            f"- HDC Index Size: {len(hdc_index['hypervectors'])} training cases"
        )
        
        return diagnosis_text, retrieval_text, attention_text, metadata_text
    
    except Exception as e:
        error_msg = f"❌ Error during inference: {str(e)}"
        return error_msg, "", "", ""

def create_demo():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="Clinical AI - Chest X-Ray Analysis",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    ) as demo:
        
        gr.Markdown("""
        # 🏥 Tri-Modal Neurosymbolic Clinical AI
        ### Chest X-Ray Classification with Four-Layer Explainability
        
        Advanced AI system combining **vision** (dual-view X-rays), **text** (radiology reports), 
        and **metadata** for accurate chest pathology detection with comprehensive explainability.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input Images")
                frontal_input = gr.Image(label="Frontal X-Ray (Required)", type="numpy", height=250)
                lateral_input = gr.Image(label="Lateral X-Ray (Optional)", type="numpy", height=250)
                
                gr.Markdown("### 📝 Clinical Report")
                report_input = gr.Textbox(
                    label="Radiology Report",
                    placeholder="Enter clinical findings and impression...\n\nExample: 'Findings: Bilateral pleural effusions. Enlarged cardiac silhouette. Impression: Congestive heart failure with cardiomegaly.'",
                    lines=8
                )
                
                submit_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                **💡 Tip:** For best results:
                - Use clear, properly oriented X-ray images
                - Include both frontal and lateral views if available
                - Provide detailed clinical findings in the report
                """)
            
            with gr.Column(scale=1):
                diagnosis_output = gr.Markdown(label="Predictions")
                
                with gr.Accordion("🔍 Similar Cases (HDC Retrieval)", open=True):
                    retrieval_output = gr.Markdown()
                
                with gr.Accordion("📊 Clinical Terms (BERT Attention)", open=True):
                    attention_output = gr.Markdown()
                
                with gr.Accordion("ℹ️ System Information", open=False):
                    metadata_output = gr.Markdown()
        
        gr.Markdown("""
        ---
        ### 📋 About This System
        
        **Architecture:** Dual-view EfficientNet-B0 (fine-tuned) + ClinicalBERT + HDC retrieval engine  
        **Training Data:** IU X-Ray dataset (2,671 studies, 8 pathology classes)  
        **Performance:** Mean AUC 88%+ | Cardiomegaly AUC 96%+  
        **Explainability:** 4 simultaneous layers (visual, textual, symbolic, analogical)
        
        **Supported Pathologies:**  
        Atelectasis • Cardiomegaly • Consolidation • Edema • Effusion • Emphysema • Pneumonia • Pneumothorax
        
        ⚠️ **Disclaimer:** This is a research prototype for educational purposes only. 
        Not intended for clinical diagnosis. Always consult qualified healthcare professionals.
        """)
        
        # Connect interface
        submit_btn.click(
            fn=predict,
            inputs=[frontal_input, lateral_input, report_input],
            outputs=[diagnosis_output, retrieval_output, attention_output, metadata_output]
        )
    
    return demo

def main():
    """Main entry point"""
    print("="*70)
    print("TRI-MODAL NEUROSYMBOLIC CLINICAL AI - LOCAL DEPLOYMENT")
    print("="*70)
    print()
    
    # Models are in the same directory as this script
    
    # Load models
    try:
        load_models()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📥 Please download the following files from Kaggle:")
        print("   1. efficientnet_fused.pth")
        print("   2. retrieval_index.pkl")
        print(f"\n   Place them in: {MODEL_DIR.absolute()}/")
        return
    
    print("\n" + "="*70)
    print("LAUNCHING GRADIO DEMO")
    print("="*70)
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )

if __name__ == "__main__":
    main()
