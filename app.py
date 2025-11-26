#!/usr/bin/env python3
"""
AI Tool Recommendation Chatbot
A Streamlit-based chatbot that recommends AI tools using semantic search
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="AI Tool Recommender",
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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .tool-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load sentence transformer model"""
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load or create sample AI tools data"""
    # Sample data structure
    data = {
        'name': [
            'ChatGPT', 'DALL-E 2', 'Midjourney', 'GitHub Copilot', 'Jasper AI',
            'Copy.ai', 'Runway ML', 'Synthesia', 'Descript', 'Otter.ai',
            'Grammarly', 'Notion AI', 'Canva AI', 'Adobe Firefly', 'Stable Diffusion'
        ],
        'category': [
            'Text Generation', 'Image Generation', 'Image Generation', 'Code Generation', 'Content Writing',
            'Content Writing', 'Video Editing', 'Video Generation', 'Audio Editing', 'Transcription',
            'Writing Assistant', 'Productivity', 'Design', 'Image Generation', 'Image Generation'
        ],
        'description': [
            'Advanced conversational AI model for natural language tasks',
            'AI system for creating images from text descriptions',
            'AI-powered art generator for creative images',
            'AI pair programmer that helps write code faster',
            'AI content platform for marketing copy and content creation',
            'AI copywriting tool for marketing and business content',
            'AI-powered video editing and generation platform',
            'AI video generation platform for creating synthetic videos',
            'All-in-one audio and video editing powered by AI',
            'AI meeting assistant for automatic transcription',
            'AI-powered writing assistant for grammar and style',
            'AI-powered workspace for note-taking and collaboration',
            'Design platform with AI-powered design tools',
            'Generative AI for creative image generation in Adobe products',
            'Open-source text-to-image model for image generation'
        ],
        'pricing': [
            'Freemium', 'Paid', 'Paid', 'Freemium', 'Paid',
            'Freemium', 'Paid', 'Paid', 'Freemium', 'Freemium',
            'Freemium', 'Freemium', 'Freemium', 'Free Beta', 'Free'
        ],
        'url': [
            'https://chat.openai.com', 'https://openai.com/dall-e-2', 'https://midjourney.com',
            'https://github.com/features/copilot', 'https://jasper.ai', 'https://copy.ai',
            'https://runwayml.com', 'https://synthesia.io', 'https://descript.com',
            'https://otter.ai', 'https://grammarly.com', 'https://notion.so',
            'https://canva.com', 'https://adobe.com/firefly', 'https://stability.ai'
        ]
    }
    return pd.DataFrame(data)

@st.cache_data
def create_embeddings(_model, texts):
    """Generate embeddings for texts"""
    embeddings = _model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).astype('float32')

def create_faiss_index(embeddings):
    """Create FAISS index"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)
    return index

def search_tools(query, model, index, df, k=5):
    """Search for relevant tools"""
    # Generate query embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    # Get results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        tool = df.iloc[idx].to_dict()
        tool['similarity_score'] = float(distance)
        results.append(tool)
    
    return results

def display_tool_card(tool):
    """Display a tool recommendation card"""
    score_color = "green" if tool['similarity_score'] > 0.5 else "orange" if tool['similarity_score'] > 0.3 else "red"
    
    st.markdown(f"""
    <div class="tool-card">
        <h3 style="color: #667eea;">🎯 {tool['name']}</h3>
        <p><strong>Category:</strong> {tool['category']} | <strong>Pricing:</strong> {tool['pricing']}</p>
        <p>{tool['description']}</p>
        <p><strong>Match Score:</strong> <span style="color: {score_color};">{tool['similarity_score']:.2%}</span></p>
        <a href="{tool['url']}" target="_blank" style="color: #667eea;">Visit Website →</a>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🤖 AI Tool Recommendation Chatbot</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Find the perfect AI tool for your needs using semantic search</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/artificial-intelligence.png", width=150)
        st.header("⚙️ Settings")
        
        top_k = st.slider("Number of recommendations", 1, 10, 5)
        
        st.markdown("---")
        st.header("📊 Statistics")
        
        # Load data
        df = load_sample_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df)}</h3>
                <p>AI Tools</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df['category'].nunique()}</h3>
                <p>Categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("📋 Categories")
        categories = df['category'].value_counts()
        for cat, count in categories.items():
            st.write(f"**{cat}:** {count} tools")
        
        st.markdown("---")
        st.header("💡 Example Queries")
        example_queries = [
            "📝 I need a tool for content writing",
            "🎨 Find me an image generation AI",
            "💻 Code assistant for developers",
            "🎥 Video editing with AI features",
            "🎯 Free tools for productivity"
        ]
        for query in example_queries:
            if st.button(query, key=query):
                st.session_state.example_query = query
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check your internet connection.")
        return
    
    # Load data and create index
    df = load_sample_data()
    combined_text = df.apply(
        lambda x: f"{x['name']} {x['category']} {x['description']} {x['pricing']}", axis=1
    ).tolist()
    
    with st.spinner("Creating search index..."):
        embeddings = create_embeddings(model, combined_text)
        index = create_faiss_index(embeddings)
    
    st.success("✅ System ready!")
    
    # Chat interface
    st.markdown("---")
    st.header("💬 Ask for AI Tool Recommendations")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle example query
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
    else:
        query = st.chat_input("🔍 What kind of AI tool are you looking for?")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching for the best AI tools..."):
                results = search_tools(query, model, index, df, k=top_k)
                time.sleep(0.5)  # For better UX
                
                response = f"I found {len(results)} AI tools that match your query:\n\n"
                st.markdown(response)
                
                for i, tool in enumerate(results, 1):
                    st.markdown(f"### {i}. {tool['name']}")
                    display_tool_card(tool)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Found {len(results)} matching tools"
                })
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🚀 <strong>AI Tool Recommendation Chatbot</strong></p>
        <p>Built with ❤️ using Streamlit, Sentence Transformers & FAISS</p>
        <p><a href="https://github.com/amalsp220/ai-tool-recommender-chatbot" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
