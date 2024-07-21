import fitz  # PyMuPDF
import re
import streamlit as st
from transformers import pipeline
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def read_pdf(file_path):
    """Read text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def summarize_text(text, max_length=150):
    """Summarize text using a pre-trained transformer model."""
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    # Split the text into sentences
    sentences = sent_tokenize(text)
    # Join sentences to form a paragraph of reasonable length
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=50, min_length=25, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

def extract_references(text):
    """Extract references from the text."""
    pattern = r"\[\d+\]"  # Pattern to match references like [1], [2], etc.
    references = re.findall(pattern, text)
    return references

# Streamlit app
st.title("Research Journal Summarizer and Reference Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_journal.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Step 1: Read the journal
    journal_text = read_pdf("uploaded_journal.pdf")
    
    st.header("Original Text")
    st.text_area("Journal Text", journal_text[:2000], height=300)  # Display first 2000 characters

    # Step 2: Summarize the content
    with st.spinner("Summarizing..."):
        summary = summarize_text(journal_text)
    
    st.header("Summary")
    st.write(summary)
    
    # Step 3: Extract references
    with st.spinner("Extracting references..."):
        references = extract_references(journal_text)

    st.header("References")
    st.write(references)

    # Optionally, delete the uploaded file after processing
    import os
    os.remove("uploaded_journal.pdf")

