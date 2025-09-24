import argparse
import os
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

# Transformers imports
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Optional: PDF/DOCX support
import PyPDF2
import docx

# Disable Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

nltk.download("punkt")


# -------------------------
# File reading functions
# -------------------------
def read_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = PyPDF2.PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif ext == ".docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return text


def split_into_sentences(text: str) -> List[str]:
    paragraphs = text.split("\n")
    sents = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            sents.extend(sent_tokenize(paragraph))
    return sents


# -------------------------
# Chunking function
# -------------------------
def chunk_sentences_by_token_count(sentences: List[str], tokenizer, max_tokens=4096, overlap=128):
    chunks = []
    cur_chunk = []
    cur_len = 0

    for sent in sentences:
        toks = len(tokenizer.encode(sent, add_special_tokens=False))

        # Case 1: sentence longer than max_tokens → split by words
        if toks > max_tokens:
            words = sent.split()
            piece = []
            piece_len = 0
            for w in words:
                w_toks = len(tokenizer.encode(w + ' ', add_special_tokens=False))
                if piece_len + w_toks > max_tokens:
                    chunks.append(' '.join(piece))
                    piece = [w]
                    piece_len = w_toks
                else:
                    piece.append(w)
                    piece_len += w_toks
            if piece:
                chunks.append(' '.join(piece))
            continue

        # Case 2: sentence fits in current chunk
        if cur_len + toks <= max_tokens:
            cur_chunk.append(sent)
            cur_len += toks
        else:
            if cur_chunk:
                chunks.append(' '.join(cur_chunk))
            # overlap
            ov = []
            ov_len = 0
            if overlap > 0:
                for s in reversed(cur_chunk):
                    s_toks = len(tokenizer.encode(s, add_special_tokens=False))
                    if ov_len + s_toks > overlap:
                        break
                    ov.insert(0, s)
                    ov_len += s_toks
            cur_chunk = ov.copy()
            cur_len = ov_len
            cur_chunk.append(sent)
            cur_len += toks

    if cur_chunk:
        chunks.append(' '.join(cur_chunk))
    return chunks


# -------------------------
# Simple extractive selection
# -------------------------
def extractive_selection(sentences: List[str], top_k=40) -> List[str]:
    # Heuristic: pick longest sentences
    sentences = sorted(sentences, key=lambda s: len(s), reverse=True)
    return sentences[:min(top_k, len(sentences))]


# -------------------------
# Main summarization function
# -------------------------
def summarize_document(file_path, use_extractive=True):
    raw = read_file(file_path)
    sentences = split_into_sentences(raw)

    # Tokenizer for chunking
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    total_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences)

    # SHORT document logic
    if total_tokens < 500:
        print("[INFO] Detected short document, using small summarization model")
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            framework="pt",  # force PyTorch
            device=-1
        )
        result = summarizer(raw, max_length=100, min_length=50, do_sample=False)
        return result[0]['summary_text']

    # LONG document logic
    if use_extractive:
        print("[INFO] Running extractive selection -> top 40 sentences")
        selected = extractive_selection(sentences, top_k=40)
    else:
        selected = sentences

    chunks = chunk_sentences_by_token_count(selected, tokenizer, max_tokens=4096)
    print(f"[INFO] Document split into {len(chunks)} chunks")

    # LED legal summarization
    led_model_name = "nsi319/legal-led-base-16384"
    led_tokenizer = AutoTokenizer.from_pretrained(led_model_name)
    led_model = AutoModelForSeq2SeqLM.from_pretrained(led_model_name)

    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[INFO] Summarizing chunk {i+1}/{len(chunks)} ...")
        input_ids = led_tokenizer(chunk, return_tensors="pt").input_ids
        output = led_model.generate(input_ids, max_new_tokens=256)
        summaries.append(led_tokenizer.decode(output[0], skip_special_tokens=True))

    final_summary = " ".join(summaries)
    print("[INFO] Producing final condensed summary")
    return final_summary


# -------------------------
# Command-line interface
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal Document Summarizer")
    parser.add_argument("-f", "--file", required=True, help="Path to document (txt, pdf, docx)")
    parser.add_argument("--no-extractive", action="store_true", help="Skip extractive pre-selection")
    args = parser.parse_args()

    summary = summarize_document(args.file, use_extractive=not args.no_extractive)
    print("\n=== FINAL SUMMARY ===\n")
    print(summary)
