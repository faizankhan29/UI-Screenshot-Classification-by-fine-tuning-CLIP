import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import clip
import torch.nn as nn
import pickle

# Load categories
@st.cache_resource
def load_categories():
    return pickle.load(open("categories.pkl", "rb"))

categories = load_categories()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load OpenAI CLIP model and preprocessing
@st.cache_resource
def load_clip_model():
    model, _ = clip.load("ViT-B/32", jit=False)
    return model.to(device)

model = load_clip_model()

# Define the fine-tuned model
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        return self.classifier(features)

# Set up the model
@st.cache_resource
def load_finetuned_model():
    num_classes = len(categories)
    model_ft = CLIPFineTuner(model, num_classes).to(device)
    model_ft.load_state_dict(torch.load('clip_finetuned.pth', map_location=device))
    model_ft.eval()
    return model_ft

model_ft = load_finetuned_model()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def process_image(image):
    """Process a single image and return prediction."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_ft(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted_label_idx = torch.max(output, 1)
        predicted_label = categories[predicted_label_idx.item()]
    return predicted_label, probabilities.tolist()

# Streamlit app
st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, probabilities = process_image(image)
    st.write(f"Predicted class: {label}")
    
    # Display top 5 predictions
    st.write("Top 5 predictions:")
    top5_prob, top5_catid = torch.topk(torch.tensor(probabilities), 5)
    for i in range(5):
        st.write(f"{categories[top5_catid[i]]}: {top5_prob[i]:.2%}")

    # Optional: Display a bar chart of top predictions
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    y_pos = range(5)
    ax.barh(y_pos, top5_prob.tolist())
    ax.set_yticks(y_pos)
    ax.set_yticklabels([categories[i] for i in top5_catid])
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Top 5 Predictions')
    
    st.pyplot(fig)

st.write("Upload an image to see the classification results!")