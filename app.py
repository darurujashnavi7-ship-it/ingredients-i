import json
from pathlib import Path

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2

APP_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Food Ingredient Identifier", page_icon="🍲", layout="wide")
st.title("🍲 Food Image Recognition & Ingredient Identification")
st.write("Upload a food photo. The CNN predicts the food category, then the app shows a curated typical ingredient list and included ingredient photos. For biryani, ingredient names come from a standard recipe list; they are not individually detected inside the image.")
st.info("Ingredient lists are typical recipe suggestions for the predicted dish. The app does not detect each ingredient individually from the photo, and recipes can vary.")

@st.cache_resource
def load_cnn():
    checkpoint = torch.load(APP_DIR / "food_classifier_finetuned.pth", map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

@st.cache_data
def load_mapping():
    with open(APP_DIR / "ingredient_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_lstm():
    checkpoint = torch.load(APP_DIR / "ingredient_lstm.pth", map_location=DEVICE, weights_only=False)
    vocab = checkpoint["vocabulary"]
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    class IngredientLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.lstm(embedded)
            return self.fc(hidden[-1])

    model = IngredientLSTM(
        len(vocab),
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim"],
        checkpoint["padding_idx"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, vocab, word_to_idx, idx_to_word, checkpoint["padding_idx"]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, model, class_names):
    x = image_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
    return class_names[index.item()], float(confidence.item()), probs.detach().cpu().tolist()

def generate_lstm_ingredients(food, lstm_bundle, max_tokens=30):
    model, vocab, word_to_idx, idx_to_word, pad_idx = lstm_bundle
    dish_token = "dish_" + food.strip().lower()
    if dish_token not in word_to_idx:
        return []
    sequence = [word_to_idx[dish_token]]
    end_idx = word_to_idx.get("<END>")
    special = {"<PAD>", "<START>", "<END>", "<UNK>"}
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([sequence], dtype=torch.long, device=DEVICE)
            next_id = int(model(x).argmax(dim=1).item())
            if next_id == end_idx or next_id == pad_idx:
                break
            sequence.append(next_id)
    return [idx_to_word[i] for i in sequence[1:] if idx_to_word[i] not in special]

# Ingredient photos are loaded from a local folder to avoid unrelated search results.
INGREDIENT_IMAGE_DIR = APP_DIR / "ingredient_images"

# Add aliases here if a recipe ingredient uses a different display name
# from the corresponding image filename.
INGREDIENT_IMAGE_ALIASES = {
    "oil or ghee": ["oil", "ghee"],
    "ghee or oil": ["ghee", "oil"],
    "basmati rice": ["basmati_rice", "rice"],
    "mint leaves": ["mint_leaves", "mint"],
    "coriander leaves": ["coriander_leaves", "coriander"],
    "red chilli powder": ["red_chilli_powder", "chilli_powder"],
    "milk solids": ["milk_solids", "khoya"],
    "wheat flour": ["wheat_flour", "flour"],
    "urad dal": ["urad_dal"],
    "fenugreek seeds": ["fenugreek_seeds"],
    "green chilli": ["green_chilli", "green_chili"],
    "soy sauce": ["soy_sauce"],
    "lemon juice": ["lemon_juice", "lemon"],
    "biryani spices": ["biryani_spices", "spices"],
    "fried onions": ["fried_onions", "onion"],
    "lemon": ["lemon", "lemon_juice"],
    "cashews": ["cashew", "cashews"],
}

def ingredient_image_candidates(ingredient):
    import re
    normalized = ingredient.strip().lower()
    candidates = INGREDIENT_IMAGE_ALIASES.get(normalized, [])
    cleaned = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    candidates = list(candidates) + [cleaned]
    # Support common image formats.
    paths = []
    for candidate in candidates:
        for extension in (".jpg", ".jpeg", ".png", ".webp"):
            paths.append(INGREDIENT_IMAGE_DIR / f"{candidate}{extension}")
    return paths

def find_ingredient_image(ingredient):
    """Return a local image path only when a matching file exists."""
    for path in ingredient_image_candidates(ingredient):
        if path.is_file():
            return path
    return None

try:
    cnn, class_names = load_cnn()
    mapping = load_mapping()
    lstm_bundle = load_lstm()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.write("Make sure the .pth and .json files are in the same folder as app.py.")
    st.stop()

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        with st.spinner("Classifying food image..."):
            food, confidence, probabilities = predict(image, cnn, class_names)
        st.subheader(f"Prediction: {food.title()}")
        st.metric("CNN confidence", f"{confidence:.1%}")
        st.caption("Confidence is the model's score, not a guarantee of correctness.")

    key = food.strip().lower()
    ingredients = mapping.get(key, [])
    st.divider()
    st.header("Ingredients")
    if ingredients:
        st.write("Typical ingredients associated with this dish:")
        cols = st.columns(3)
        for i, ingredient in enumerate(ingredients):
            with cols[i % 3]:
                st.markdown(f"**{ingredient}**")
                image_path = find_ingredient_image(ingredient)
                if image_path:
                    st.image(str(image_path), caption=ingredient, use_container_width=True)
                else:
                    st.caption("No local photo added for this ingredient yet.")
    else:
        st.warning("No ingredient mapping found for this category.")


else:
    st.markdown("### How to use")
    st.markdown("1. Upload a food photo.  \n2. Review the predicted food and confidence.  \n3. View the typical ingredient list and available ingredient images.")
