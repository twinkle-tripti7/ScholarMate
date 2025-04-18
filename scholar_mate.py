import streamlit as st
import fitz  # PyMuPDF
import tempfile
import arxiv
import requests
from transformers import pipeline, AutoTokenizer

# Load summarizer and tokenizer from Hugging Face
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()
tokenizer = load_tokenizer()

# Streamlit UI
st.title("📚 ScholarMate - Research Summarizer")
option = st.selectbox("Choose an option:", ["Search for Papers", "Upload a PDF"])

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if option == "Search for Papers":
    topic = st.text_input("Enter the topic to search for research papers:")

    if topic:
        search_results = arxiv.Search(
            query=topic,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )

        st.subheader(f"📄 Top results for '{topic}':")
        for result in search_results.results():
            title = result.title
            authors = [author.name for author in result.authors]
            pdf_url = result.pdf_url

            st.markdown(f"### {title}")
            st.markdown(f"**Authors:** {', '.join(authors)}")
            st.markdown(f"[🔗 PDF Link]({pdf_url})")

            if st.button(f"Summarize Paper: {title}"):
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    pdf_text = extract_text_from_pdf(response.content)

                    # Truncate input text for model
                    inputs = tokenizer(pdf_text, return_tensors="pt", truncation=True, max_length=1024)
                    trimmed_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

                    try:
                        summary_output = summarizer(trimmed_text, max_length=300, min_length=80, do_sample=False)
                        summary_text = summary_output[0]['summary_text']
                        st.subheader("📄 AI Summary:")
                        st.write(summary_text)
                    except Exception as e:
                        st.error(f"❌ Error during summarization: {e}")

                    st.subheader("📚 Citation (APA):")
                    citation = f"{', '.join(authors)}. ({result.published.date()}). {title}. Retrieved from {pdf_url}"
                    st.code(citation)
                else:
                    st.error("Failed to fetch the paper.")

elif option == "Upload a PDF":
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        if full_text.strip() == "":
            st.error("Couldn't extract text from PDF.")
        else:
            st.success("PDF uploaded and text extracted successfully!")
            st.subheader("📑 Raw Extracted Text")
            st.text_area("Preview (editable)", value=full_text[:3000], height=300)

            if st.button("Summarize PDF"):
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
                trimmed_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

                try:
                    summary_output = summarizer(trimmed_text, max_length=300, min_length=80, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    st.subheader("📄 AI Summary:")
                    st.write(summary_text)
                except Exception as e:
                    st.error(f"❌ Error during summarization: {e}")
