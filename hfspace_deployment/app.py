"""
Autism Information Chatbot - Streamlit Version
RAG system using Qwen2.5-1.5B-Instruct + FAISS + BGE embeddings
"""

import gc
import pickle
import torch
import faiss
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH  = "./lora_weights"
FAISS_INDEX_PATH = "./faiss_index.bin"
CHUNKS_PATH      = "./pdf_chunks.pkl"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

REFUSAL_STRING = (
    "I do not understand this question, and the topic you mentioned "
    "is not present in the provided PDF corpus."
)

# Topics that should ALWAYS be refused
FORBIDDEN_TOPICS = [
    'burger', 'pizza', 'pasta', 'recipe', 'cook', 'bake', 'ingredient',
    'kitchen', 'food', 'restaurant', 'meal', 'breakfast', 'lunch', 'dinner',
    'capital of', 'president of', 'prime minister', 'population of',
    'invented', 'inventor', 'discovered', 'founded',
    'tire', 'tyre', 'wheel', 'car', 'vehicle', 'automobile',
    'stock price', 'share price', 'market cap', 'dollar', 'euro', 'bitcoin',
    'super bowl', 'world cup', 'olympics', 'championship', 'sports',
    'weather', 'temperature', 'forecast', 'rain', 'sunny',
    'movie', 'film', 'actor', 'actress', 'celebrity',
    'song', 'music', 'band', 'album', 'singer',
    'how to fix', 'how to build', 'how to make', 'how to cook',
    'how to install', 'how to change', 'how to repair',
]

# Only these topics should be answered
AUTISM_KEYWORDS = [
    'autism', 'asd', 'asperger', 'spectrum', 'autistic',
    'developmental', 'neurodevelopmental', 'disorder',
    'behavior', 'behaviour', 'child', 'children', 'infant', 'toddler',
    'treatment', 'therapy', 'intervention', 'management',
    'symptom', 'sign', 'feature', 'characteristic',
    'diagnosis', 'diagnose', 'screen', 'assessment',
    'cause', 'etiology', 'genetic', 'risk factor',
    'kanner', 'rimland', 'ivar lovaas',
    'sensory', 'communication', 'social', 'repetitive',
    'language', 'speech', 'nonverbal', 'eye contact',
    'aba', 'applied behavior', 'early intervention',
    'gfcf', 'gluten', 'casein', 'diet',
    'prevalence', 'epidemiology', 'incidence',
    'sleep', 'anxiety', 'adhd', 'comorbid',
    'iep', 'special education', 'inclusion',
    'stimming', 'meltdown', 'sensory processing',
]

def is_autism_related(question: str) -> bool:
    """Check if question is about autism/related topics."""
    q = question.lower()
    return any(keyword in q for keyword in AUTISM_KEYWORDS)

def is_forbidden(question: str) -> bool:
    """Check if question is clearly off-topic."""
    q = question.lower()
    return any(topic in q for topic in FORBIDDEN_TOPICS)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models, please wait (this may take a few minutes)...")
def load_all():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    if Path(LORA_MODEL_PATH).exists():
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    gc.collect()

    return embedder, index, chunks, tokenizer, model


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(question, embedder, index, chunks, k=5):
    instruction = "Represent this sentence for searching relevant passages: "
    q_emb = embedder.encode(
        [instruction + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    _, indices = index.search(q_emb, k)
    return " ".join(chunks[i] for i in indices[0])


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
def ask_qwen(question, context, tokenizer, model):
    # Hard refusal: forbidden topic
    if is_forbidden(question):
        return REFUSAL_STRING

    # Hard refusal: not autism-related at all
    if not is_autism_related(question):
        return REFUSAL_STRING

    system_prompt = f"""You are a STRICT assistant that ONLY answers questions about autism spectrum disorder using the Context below.

ABSOLUTE RULES — NO EXCEPTIONS:
1. You ONLY use information from the Context below. NEVER use your own knowledge.
2. If the Context does not contain a clear answer, respond EXACTLY with this sentence and nothing else:
   "{REFUSAL_STRING}"
3. Do NOT answer questions about food, cooking, geography, sports, weather, or any non-autism topic.
4. Do NOT make up or infer information not in the Context.
5. If you are even slightly unsure, use the refusal message.

Context:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return response


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Autism Information Chatbot",
    page_icon="🧩",
    layout="centered",
)

st.title("🧩 Autism Information Chatbot")
st.caption(
    "Ask questions about Autism Spectrum Disorder. "
    "Answers are grounded exclusively in research PDFs."
)
st.warning(
    "⚠️ This chatbot is for **informational purposes only** and is "
    "not a substitute for professional medical advice.",
    icon="⚠️",
)

embedder, index, chunks, tokenizer, model = load_all()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.header("💡 Example Questions")
    examples = [
        "What is autism?",
        "What are the main symptoms of autism?",
        "How is autism diagnosed?",
        "What treatments are available?",
        "What are early signs of autism in children?",
        "What is Asperger's syndrome?",
        "What causes autism?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask a question about autism...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs and generating answer..."):
            context  = retrieve(user_input, embedder, index, chunks)
            response = ask_qwen(user_input, context, tokenizer, model)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})