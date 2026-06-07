import streamlit as st
import os
from src.ingest import process_and_store_pdf
from src.pipeline import run_rag_pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="InsightDocs // AI Engine",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Premium CSS (With Explicit Placeholder Fixes)
st.markdown("""
    <style>
        /* Main background and font adjustments */
        .stApp {
            background-color: #0E1117;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Premium custom heading styling */
        h1 {
            color: #FFFFFF !important;
            font-weight: 800 !important;
            letter-spacing: -1px !important;
            font-size: 2.5rem !important;
            margin-bottom: 5px !important;
        }
        h3 {
            color: #A0AEC0 !important;
            font-weight: 400 !important;
            font-size: 1.1rem !important;
            margin-bottom: 25px !important;
        }
        
        /* Modernized Text input boxes */
        .stTextInput input {
            background-color: #1A1F2C !important;
            color: #FFFFFF !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            transition: all 0.3s ease;
        }
        .stTextInput input:focus {
            border-color: #10B981 !important; /* Emerald Accent */
            box-shadow: 0 0 0 1px #10B981 !important;
        }
        
        /* CRITICAL FIX: Force placeholder text to be visible against dark background */
        .stTextInput input::placeholder {
            color: #718096 !important;
            opacity: 1 !important; /* Firefox override */
        }
        .stTextInput input::-webkit-input-placeholder {
            color: #718096 !important;
        }
        .stTextInput input::-moz-placeholder {
            color: #718096 !important;
            opacity: 1 !important;
        }
        .stTextInput input:-ms-input-placeholder {
            color: #718096 !important;
        }
        
        /* Custom Primary Action Button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
            width: 100% !important;
            letter-spacing: 0.5px !important;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
        }
        
        /* Styled Expander / Dropdowns for Context sources */
        .streamlit-expanderHeader {
            background-color: #1A1F2C !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            color: #E2E8F0 !important;
        }
        .streamlit-expanderContent {
            background-color: #111622 !important;
            border-left: 3px solid #10B981 !important;
            padding: 15px !important;
            color: #CBD5E0 !important;
            font-size: 0.95rem !important;
        }
        
        /* Custom cards for answers */
        .answer-card {
            background-color: #1A1F2C;
            border-left: 4px solid #10B981;
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Clean up sidebar background blending */
        [data-testid="stSidebar"] {
            background-color: #0B0E14 !important;
            border-right: 1px solid #1A1F2C !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Main Header Section
st.markdown("<h1>InsightDocs <span style='color: #10B981;'>//</span></h1>", unsafe_allow_html=True)
st.markdown("<h3>Context-Aware Vector Search & Document Intelligence</h3>", unsafe_allow_html=True)

# 4. Sidebar / Control Center Interface
st.sidebar.markdown("<h2 style='color: white; font-size: 1.3rem; font-weight: 700; margin-top: 20px;'>CONTROL PANEL</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #718096; font-size: 0.85rem;'>Upload any PDF to extract semantic vectors instantly.</p>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Choose Document",
    type=["pdf"],
    label_visibility="collapsed"
)

TEMP_DIR = "data"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

index_ready = False

if uploaded_file is not None:
    file_path = os.path.join(TEMP_DIR, "active_document.pdf")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        with st.sidebar.spinner("Generating structural embeddings..."):
            try:
                process_and_store_pdf(file_path, vector_store_path="faiss_index")
                st.session_state["current_file"] = uploaded_file.name
                st.sidebar.success(f"✓ {uploaded_file.name} Indexed")
            except Exception as e:
                st.sidebar.error(f"Ingestion Fail: {str(e)}")
                
    index_ready = True
else:
    if os.path.exists("faiss_index"):
        st.sidebar.info("● Systems Nominal (Cached database active)")
        index_ready = True
    else:
        st.info("⚡ System idle. Please feed a PDF document into the control panel sidebar to spin up the vector map.")

# 5. Core Chat Query Interface
if index_ready:
    # Explicit placeholder text layout
    user_query = st.text_input(
        "Search Query",
        placeholder="Ask anything about the active document (e.g., requirements, dates)...",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the execution button smoothly
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_clicked = st.button("EXECUTE ANALYSIS")
        
    if submit_clicked:
        if user_query.strip() == "":
            st.warning("Query cannot be blank.")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("Executing similarity search & routing matrices..."):
                try:
                    response = run_rag_pipeline(user_query)
                    
                    # Display the Main Answer styled inside a clean card block
                    st.markdown("<p style='color: #A0AEC0; font-weight: 600; font-size: 0.9rem; margin-bottom: -5px; letter-spacing: 0.5px;'>GENERATED RESPONSE</p>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="answer-card">
                            <span style="color: #F7FAFC; font-size: 1.05rem; line-height: 1.6;">{response["answer"]}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Sources
                    st.markdown("<p style='color: #A0AEC0; font-weight: 600; font-size: 0.9rem; margin-bottom: 10px; letter-spacing: 0.5px;'>VERIFIED CONTEXT VECTOR BLOCKS</p>", unsafe_allow_html=True)
                    for i, doc in enumerate(response["context"]):
                        page_num = doc.metadata.get("page", "Unknown")
                        with st.expander(f"Block 0{i+1} — Reference Metadata [Page {page_num}]"):
                            st.write(doc.page_content)
                            
                except Exception as e:
                    st.error(f"Pipeline Execution Failed: {str(e)}")

# Minimalist Terminal Style Footer
st.markdown("<br><br><br><p style='text-align: center; color: #4A5568; font-size: 0.75rem; letter-spacing: 1px;'>SYSTEM ENVIRONMENT: PRODUCTION // SECURE NODE</p>", unsafe_allow_html=True)