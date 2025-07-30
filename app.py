import streamlit as st
from utils.file_loader import load_files
from utils.chunker import chunk_sections
from utils.faiss_handler import build_faiss_index, get_top_chunks
from utils.retriever import generate_response
from prompts.chain_of_thought import cot_prompt
from utils.evaluation import evaluate_predictions
import pandas as pd
import re


st.set_page_config(page_title="RAG App with Groq", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stTextInput {
        margin-bottom: 1rem;
    }
    .stButton button {
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        background-color: #f0f2f6;
        border: 1px solid #d6d9df;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }
    .stButton button:hover {
        background-color: #e4e7eb;
    }
    .stTable {
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🧠 BUSINESS CASE STUDY ON FLIPKART" )

uploaded_files = st.file_uploader("Upload 10 documents", type=['txt', 'pdf', 'json', 'xml', 'xlsx'], accept_multiple_files=True)

query = st.text_input("Ask a question to your knowledge base:")

if uploaded_files and query:
    with st.spinner("Processing..."):

        # STEP 1: Load file content
        documents = load_files(uploaded_files)

        # STEP 2: Detect filename-based queries
        filename_question_match = re.search(r"what does (.*?) say", query.lower())
        if filename_question_match:
            keyword = filename_question_match.group(1).strip().replace(" ", "").lower()

            matched_docs = [
                doc for doc in documents
                if keyword in doc["filename"].lower().replace(" ", "")
            ]

            if matched_docs:
                st.markdown(f"📄 Found content from **{matched_docs[0]['filename']}**")
                st.write(matched_docs[0]['content'][:2000])

                final_prompt = f"""Use the following document content to answer the question:

                Content from {matched_docs[0]['filename']}:
                {matched_docs[0]['content'][:3000]}

                Question: {query}
                Answer:"""
                
                answer = generate_response(final_prompt)

                st.markdown("### 💬 Answer")
                st.write(answer)
            else:
                st.warning(f"No document found matching **{keyword}**")

        else:
            # STEP 3: Chunking logic
            if any(
                isinstance(doc, dict) and
                isinstance(doc.get("content", ""), str) and
                "Year:" in doc["content"] and "Revenue:" in doc["content"]
                for doc in documents
            ):
                chunks = documents  # Already structured (Excel)
            else:
                chunks = chunk_sections(documents)
            # STEP 4: Build FAISS index
            index, chunk_texts = build_faiss_index(chunks)

            # STEP 5: Retrieve top chunks
            relevant_chunks = get_top_chunks(index, chunk_texts, query, top_k=5)

            # STEP 6: Generate prompt with Chain-of-Thought
            final_prompt = cot_prompt(query, relevant_chunks)
            answer = generate_response(final_prompt)

            st.markdown("### 💬 Answer")
            st.write(answer)

            # ✅ Show retrieved chunk sources before generating answer
            st.markdown("### 📄 Top Chunks Used")
            for idx, chunk in enumerate(relevant_chunks, 1):
                    filename = chunk.get("metadata", {}).get("filename", "Unknown")
                    st.markdown(f"**Chunk {idx}: {filename}**")
                    st.code(chunk.get("content", "")[:1000])


                # if isinstance(chunk, dict):

                #     st.markdown(f"**Chunk {idx}: {chunk.get('filename', 'Unknown')}**")
                #     st.code(chunk.get("content", "")[:1000])  # Show first 1000 chars
                # else:
                #     st.markdown(f"**Chunk {idx}**")
                #     st.code(chunk[:1000])



# -------------------
# Evaluation Section
# -------------------

ground_truth_data = {
    "What is Flipkart?": "Flipkart is an Indian e-commerce company.",
    "When was Flipkart founded?": "Flipkart was founded in October 2007.",
    "Where was Flipkart founded?": "Flipkart was founded in Bangalore.",
    "Who founded Flipkart?": "Flipkart was founded by Sachin Bansal and Binny Bansal.",
    "What was Flipkart's initial focus?": "Initially, Flipkart focused solely on selling books online."
}

st.markdown("## 📊 Evaluate Your RAG Model")

if not uploaded_files:
    st.warning("⚠️ Please upload a file first — evaluation uses retrieved chunks.")

elif st.button("Evaluate Model"):
    with st.spinner("Running evaluation with live model answers..."):
        model_answers = {}

        for question in ground_truth_data:
            relevant_chunks = get_top_chunks(index, chunk_texts, question, top_k=5)
            prompt = cot_prompt(question, relevant_chunks)
            answer = generate_response(prompt).strip()
            model_answers[question] = answer

        results = evaluate_predictions(ground_truth_data, model_answers)

    st.success("✅ Evaluation complete!")

    metric_pairs = [(k, v) for k, v in results.items() if k != "Model Used"]
    df = pd.DataFrame(metric_pairs, columns=["Metric", "Score"])
    st.markdown("### 📊 Evaluation Metrics")
    st.table(df)
    explanation_map = {
    "ROUGE-1": (
        "Measures the overlap of **unigrams** (individual words) between the model’s answer and the expected answer.\n\n"
        "- High score (≥ 0.75): Excellent word overlap.\n"
        "- Moderate score (0.50–0.75): Good overlap, minor wording differences.\n"
        "- Low score (< 0.50): Significant wording mismatch or missing terms."
    ),
    "ROUGE-L": (
        "Focuses on the **longest common subsequence** between predicted and actual answers, capturing structure/order.\n\n"
        "- High score (≥ 0.75): Structure and phrasing are highly aligned.\n"
        "- Moderate score (0.50–0.75): Generally aligned but with reordered phrases.\n"
        "- Low score (< 0.50): Different phrasing or fragmented structure."
    ),
    "Cosine Similarity": (
        "Measures the **semantic similarity** between the model’s answer and the expected one.\n\n"
        "- High score (≥ 0.85): Strong semantic match.\n"
        "- Moderate score (0.60–0.85): Similar meaning with different wording.\n"
        "- Low score (< 0.60): Weak or incorrect meaning."
    ),
    "F1_SCORE": (
        "Harmonic mean of **precision** and **recall**, reflecting correctness and completeness.\n\n"
        "- High score (≥ 0.75): Mostly correct and complete.\n"
        "- Moderate score (0.50–0.75): Some errors or omissions.\n"
        "- Low score (< 0.50): Many incorrect or missing pieces."
    ),
    "Accuracy": (
        "Measures whether the predicted answer **exactly matches** the expected answer.\n\n"
        "- Score of 1.00: Perfect match.\n"
        "- Score of 0.00: No exact matches.\n"
        "- This metric is harsh — even small variations will reduce it."
    )
}
   
        
    def quality_level(score):
        if score >= 0.75:
            return "Excellent"
        elif 0.5 <= score < 0.75:
            return "Good"
        elif 0.3 <= score < 0.5:
            return "Fair"
        else:
            return "Poor"

    for metric, score in results.items():
        if metric == "Model Used":
            continue
        
        explanation = explanation_map.get(metric, "No explanation available.")
        level = quality_level(score)

        st.markdown(f"### {metric}: {score:.2f} ({level})")
        st.markdown(f"**Your score:** `{score:.2f}` – Quality: **{level}**")
        st.markdown(explanation)


