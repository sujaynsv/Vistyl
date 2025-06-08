import streamlit as st
import pandas as pd
import numpy as np
import clip
import torch
import faiss
from PIL import Image
from io import BytesIO
import requests
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="Fashion Visual Search & Outfit Recommender")

# Load CLIP model
@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_clip_model()

# Load cleaned data and precomputed embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("fashion_products_clean.csv")
    vectors = np.load("image_vectors.npy").astype("float32")
    return df, vectors

fashion_df, image_vectors = load_data()

# Build FAISS index
@st.cache_resource
def build_faiss_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

faiss_index = build_faiss_index(image_vectors)

# Search similar products
def search_similar_products(image, k=5):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_vec = model.encode_image(image_tensor).cpu().numpy().astype("float32")
    D, I = faiss_index.search(query_vec, k)
    return fashion_df.iloc[I[0]]

# Outfit recommender
def get_outfit_recommendations(product_row, k=5):
    dept = product_row['department_id']
    brand = product_row['brand']
    recommendations = fashion_df[
        (fashion_df['department_id'] == dept) & (fashion_df['brand'] != brand)
    ]
    return recommendations.sample(min(k, len(recommendations)))

# --- Streamlit UI ---
st.title("Fashion Visual Search & Outfit Recommender")
st.write("Upload a fashion item image to find visually similar products and outfit suggestions.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Finding similar items..."):
        similar_items = search_similar_products(image, k=5)

    st.subheader("Visually Similar Products")
    for _, row in similar_items.iterrows():
        st.image(row['feature_image_s3'], caption=row['product_name'])
        st.write(f"**Brand:** {row['brand']}, **Price:** {row['selling_price']}")
        st.markdown(f"[View Product]({row['pdp_url']})")
        st.markdown("---")

    st.subheader("Outfit Suggestions")
    outfit_df = get_outfit_recommendations(similar_items.iloc[0], k=3)
    for _, row in outfit_df.iterrows():
        st.image(row['feature_image_s3'], caption=row['product_name'])
        st.write(f"**Brand:** {row['brand']}, **Price:** {row['selling_price']}")
        st.markdown(f"[View Product]({row['pdp_url']})")
        st.markdown("---")
