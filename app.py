import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
@st.cache_resource
def load_model():
    model_name = "D:\stage PFE 2025\les models à garder bien entrainer\qwen2.5 0.5B\saved_model"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Utiliser float32 sur CPU
        
    ).to('cpu')

    # Vérifier et désactiver l'attention par fenêtre glissante si nécessaire
    if hasattr(model.config, 'use_sliding_window_attention'):
        model.config.use_sliding_window_attention = False

    # Vérifier le type d'attention
    if hasattr(model.config, 'attention_type'):
        model.config.attention_type = "softmax"
    return tokenizer, model

tokenizer, model = load_model()

st.title("🚀 Générateur de texte avec mon modèle")

prompt = st.text_area("Entrer le texte initial :", value="")

max_length = st.slider("Longueur maximale du texte généré :", min_value=10, max_value=200, value=50)
temperature = st.slider("Température (créativité) :", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

if st.button("Générer le texte"):
    if prompt.strip() == "":
        st.warning("Veuillez entrer un texte initial.")
    else:
        with st.spinner("Génération en cours..."):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            st.success("🎉 Texte généré :")
            st.write(output)
"""

@st.cache_resource
def load_model():
    model_name = "D:\stage PFE 2025\les models à garder bien entrainer\qwen2.5 0.5B\saved_model"
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Charger le modèle en forçant l'utilisation du CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    ).to('cpu')

    # Vérifier et désactiver l'attention par fenêtre glissante si nécessaire
    if hasattr(model.config, 'use_sliding_window_attention'):
        model.config.use_sliding_window_attention = False

    # Vérifier le type d'attention et définir "softmax" si applicable
    if hasattr(model.config, 'attention_type'):
        model.config.attention_type = "softmax"
    
    return tokenizer, model

# Charger le modèle et le tokenizer
tokenizer, model = load_model()

# Interface Streamlit
st.title("🚀 Générateur de texte avec mon modèle Qwen2.5-0.5B")

# Zone de texte pour le prompt
prompt = st.text_area("Entrer le texte initial :", value="")

# Paramètres de génération
max_length = st.slider("Longueur maximale du texte généré :", min_value=10, max_value=200, value=50)
temperature = st.slider("Température (créativité) :", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

if st.button("Générer le texte"):
    if prompt.strip() == "":
        st.warning("Veuillez entrer un texte initial.")
    else:
        with st.spinner("Génération en cours..."):
            try:
                # Encoder le prompt
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cpu')
                
                # Génération du texte
                output_ids = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Décoder et afficher le résultat
                output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                st.success("🎉 Texte généré :")
                st.write(output)

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la génération : {e}")
