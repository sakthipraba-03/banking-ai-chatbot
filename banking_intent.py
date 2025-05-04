import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define model directory 
MODEL_DIR = "./intent_model"

# STEP 2: Define manual label mapping 
label_map = {
    0: "banking",
    1: "non_banking"
}

# Load tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  

# Define inference function 
def predict_intent(text, temperature=0.1):   
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply temperature
        logits = logits / temperature  
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    predicted_label = label_map[predicted_class]
    return predicted_label, confidence

# Test with a sample query 
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        label, conf = predict_intent(user_input, temperature=0.1)  
        print(f"Predicted Intent: {label} (Confidence: {conf:.2f})")