import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import sys
import os
import numpy as np



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/xray_model.pth'



def load_model():
    """Load the trained model"""
    model = models.resnet50(pretrained=False)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model



def predict(image_path, model):
    """Predict class for a single X-ray image"""
    
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs_tensor = torch.softmax(outputs, dim=1)[0]
        probabilities = probs_tensor.cpu().numpy()
    

    probs_corrected = probabilities.copy()
    
    predicted_class_idx = probs_corrected.argmax()
    confidence = probs_corrected[predicted_class_idx]
    
    classes = ['COVID-19', 'Normal', 'Pneumonia']
    
    return {
        'class': classes[predicted_class_idx],
        'confidence': float(confidence),
        'probabilities': {
            'COVID-19': float(probs_corrected[0]),
            'Normal': float(probs_corrected[1]),
            'Pneumonia': float(probs_corrected[2])
        }
    }



def generate_heatmap(image_path, model):
    """Generate Grad-CAM heatmap showing which parts the model focuses on"""
    try:
        import cv2
    except ImportError:
        print("⚠️  opencv-python not installed. Install with: pip install opencv-python")
        return None
    
    # Load original image
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error reading image: {image_path}")
        return None
    img_cv = cv2.resize(img_cv, (224, 224))
    
    # Prepare tensor
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad = True
    
    # Forward pass
    output = model(img_tensor)
    predicted_class = output.argmax(dim=1).item()
    
    # Backward pass to get gradients
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][predicted_class] = 1
    output.backward(gradient=one_hot)
    
    # Extract and process gradients
    gradients = img_tensor.grad[0].mean(dim=0)
    gradients = gradients.detach().cpu().numpy()
    
    # Normalize gradients
    gradients = np.maximum(gradients, 0)
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
    gradients = cv2.resize(gradients, (224, 224))
    gradients = (gradients * 255).astype(np.uint8)
    
    # Apply colormap (red = high attention)
    heatmap = cv2.applyColorMap(gradients, cv2.COLORMAP_JET)
    
    # Blend heatmap with original image
    result = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    
    # Save heatmap image
    base_name = os.path.basename(image_path)
    heatmap_filename = f"heatmap_{base_name.replace('.', '_')}.png"
    cv2.imwrite(heatmap_filename, result)
    
    print(f"✓ Heatmap saved to: {heatmap_filename}")
    return heatmap_filename



def generate_llm_report(prediction_result, image_path=""):
    """Generate medical report using Claude AI"""
    try:
        import anthropic
    except ImportError:
        print("\n❌ anthropic library not installed!")
        print("Install with: pip install anthropic\n")
        return None
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("\n" + "="*80)
        print("  ANTHROPIC_API_KEY NOT SET")
        return None
    
    try:
        print("✓ Connecting to Claude AI...")
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""
You are an expert radiologist assistant. Based on the following AI analysis of a chest X-ray, 
generate a professional and detailed medical report.

=== AI ANALYSIS RESULTS ===
Predicted Diagnosis: {prediction_result['class']}
Confidence Score: {prediction_result['confidence']*100:.1f}%

Probability Breakdown:
- COVID-19: {prediction_result['probabilities']['COVID-19']*100:.1f}%
- Normal: {prediction_result['probabilities']['Normal']*100:.1f}%
- Pneumonia: {prediction_result['probabilities']['Pneumonia']*100:.1f}%

=== TASK ===
Generate a comprehensive radiologist report including:

1. **CLINICAL HISTORY** (assume routine chest X-ray screening)
2. **TECHNIQUE** (Note: standard chest X-ray analysis)
3. **FINDINGS** - Detailed description of what the AI detected
4. **IMPRESSION** - Summary of the diagnosis with confidence assessment
5. **RECOMMENDATIONS** - Next steps and follow-up actions

Keep the report professional, concise (300-400 words), and suitable for a medical record.
Include appropriate medical disclaimers about AI-assisted analysis.
"""
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        print("✓ Report generated successfully!\n")
        return message.content[0].text
    
    except anthropic.AuthenticationError:
        print("\n❌ Authentication failed - invalid API key!")
        print("Check your API key at: https://console.anthropic.com\n")
        return None
    except anthropic.RateLimitError:
        print("\n⚠️  Rate limit exceeded - try again in a moment\n")
        return None
    except anthropic.APIError as e:
        print(f"\n❌ API Error: {str(e)}\n")
        return None
    except Exception as e:
        print(f"\n⚠️  Error generating report: {str(e)}\n")
        return None



def save_report_to_file(image_path, prediction_result, llm_report=None):
    """Save prediction and report to a text file"""
    
    base_name = os.path.basename(image_path)
    report_filename = f"report_{base_name.replace('.', '_')}.txt"
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("X-RAY ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("FILE INFORMATION:")
    report_lines.append(f"  Image: {image_path}")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("AI PREDICTION RESULTS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Predicted Diagnosis: {prediction_result['class']}")
    report_lines.append(f"Confidence Score: {prediction_result['confidence']*100:.2f}%")
    report_lines.append("")
    report_lines.append("Detailed Probabilities:")
    report_lines.append(f"  - COVID-19:  {prediction_result['probabilities']['COVID-19']*100:.2f}%")
    report_lines.append(f"  - Normal:    {prediction_result['probabilities']['Normal']*100:.2f}%")
    report_lines.append(f"  - Pneumonia: {prediction_result['probabilities']['Pneumonia']*100:.2f}%")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("MEDICAL REPORT (Generated by Claude AI)")
    report_lines.append("="*80)
    report_lines.append("")
    
    if llm_report:
        report_lines.append(llm_report)
    else:
        report_lines.append("[LLM Report Not Generated - See console output for setup instructions]")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("DISCLAIMER")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("This report is generated using AI-assisted analysis. It is intended as a")
    report_lines.append("supportive tool for medical professionals and should NOT be used as a")
    report_lines.append("standalone diagnostic tool.")
    report_lines.append("")
    report_lines.append("IMPORTANT:")
    report_lines.append("- Final diagnosis must be confirmed by a qualified radiologist")
    report_lines.append("- This system is NOT FDA-approved for clinical use")
    report_lines.append("- AI predictions may contain errors - always seek professional medical advice")
    report_lines.append("- Do not use for independent clinical decision-making")
    report_lines.append("- This is a research/prototype system only")
    report_lines.append("")
    report_lines.append("For official medical diagnosis, please consult with a licensed radiologist.")
    
    report_content = "\n".join(report_lines)
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_filename



if __name__ == '__main__':
    print("Loading model...")
    model = load_model()
    print("✓ Model loaded successfully!\n")
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py data/test/COVID/image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    print(f"Analyzing X-ray: {image_path}")
    print("-" * 80)
    
    result = predict(image_path, model)
    
    # Display results
    print(f"\n🫁 PREDICTED DIAGNOSIS: {result['class']}")
    print(f"📊 CONFIDENCE SCORE: {result['confidence']*100:.2f}%")
    print(f"\n📈 PROBABILITY BREAKDOWN:")
    for class_name in ['COVID-19', 'Normal', 'Pneumonia']:
        prob = result['probabilities'][class_name]
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {class_name:12} [{bar}] {prob*100:6.2f}%")
    
    # Generate Grad-CAM heatmap
    print("\n" + "="*80)
    print("GENERATING GRAD-CAM HEATMAP...")
    print("="*80 + "\n")
    heatmap_file = generate_heatmap(image_path, model)
    
    # Generate LLM report
    print("\n" + "="*80)
    print("GENERATING DETAILED MEDICAL REPORT...")
    print("="*80 + "\n")
    
    llm_report = generate_llm_report(result, image_path)
    
    if llm_report:
        print(llm_report)
    
    # Save report to file
    print("\n" + "="*80)
    report_file = save_report_to_file(image_path, result, llm_report)
    print(f"✓ Report saved to: {report_file}")
    if heatmap_file:
        print(f"✓ Heatmap saved to: {heatmap_file}")
    print("="*80)