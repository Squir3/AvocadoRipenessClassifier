import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Konfiguracja strony
st.set_page_config(page_title="Klasyfikator Dojrzałości Awokado", layout="centered")

# Tytuł
st.title("🥑 Klasyfikator dojrzałości awokado🥑")
st.write("Prześlij zdjęcie awokado, a model określi jego stopień dojrzałości:")
st.write("1 — Niedojrzałe")
st.write("2 — Faza przejściowa (Breaking)")
st.write("3 — Dojrzałe (pierwszy etap)")
st.write("4 — Dojrzałe (maksimum)")
st.write("5 — Przejrzałe")

# Mapa etykiet
label_map = {
    0: "1 — Niedojrzałe",
    1: "2 — Faza przejściowa (Breaking)",
    2: "3 — Dojrzałe (pierwszy etap)",
    3: "4 — Dojrzałe (maksimum)",
    4: "5 — Przejrzałe"
}

# Wczytanie modelu
@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load("resnet_avocado.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Transformacje obrazu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Upload zdjęcia
uploaded_file = st.file_uploader("📷 Wybierz zdjęcie awokado", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Wczytane zdjęcie", use_container_width=True)

    # Przetwarzanie i predykcja
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, 1).item()

    st.subheader("🔍 Wynik klasyfikacji:")
    st.success(f"{label_map[prediction]}")
