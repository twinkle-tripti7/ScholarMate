# ScholarMate - Research Summarizer

**ScholarMate** is a web application designed to assist researchers and students by summarizing academic papers and generating citations. The app integrates several functionalities to streamline the research process, including the ability to extract text from PDF papers, summarize them using state-of-the-art language models, and generate proper citations in various formats.

## Features
- **Search for Research Papers**: Search for papers on [arXiv](https://arxiv.org/) based on a given topic.
- **PDF Upload & Text Extraction**: Upload research papers (PDFs), extract the text, and generate summaries.
- **AI-Powered Summarization**: Summarize papers using Hugging Face's DistilBART model.
- **Citation Generation**: Generate citations in APA format.
- **User-Friendly Interface**: Built with Streamlit for an easy-to-use web interface.

## Demo
[Link to Demo (if applicable)](your-demo-link)

## Technologies Used
- **Streamlit**: For creating the interactive web application.
- **Hugging Face (DistilBART)**: For paper summarization.
- **OpenAI GPT-3**: For additional text summarization and citation generation.
- **PyMuPDF (fitz)**: For extracting text from PDF documents.
- **arXiv API**: To search for academic papers from arXiv.

## Installation

### Prerequisites
Make sure you have Python 3.x installed. You also need to install the necessary libraries.

### Steps
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/ScholarMate.git
    cd ScholarMate
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Setup Environment Variables**:
    ```bash
    HUGGINGFACE_TOKEN=your-huggingface-token
    ```

5. **Run the Application**:
    ```bash
    streamlit run app.py
    ```


