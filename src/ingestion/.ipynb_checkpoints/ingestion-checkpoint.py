import json
import pandas as pd
import re
import spacy
from pathlib import Path
from typing import List, Dict

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_json_faq(filepath: Path) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    faqs = []
    for cat in data.get('categories', []):
        category = cat.get('category', '').strip()
        for qa in cat.get('questions', []):
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            faqs.append({'category': category, 'question': question, 'answer': answer})
    return faqs

def load_excel_qa(filepath: Path) -> List[Dict]:
    xls = pd.ExcelFile(filepath)
    faqs = []
    question_starters = re.compile(r"^(what|why|how|is|are|can|could|should|do|does|who|when|where)\b", re.IGNORECASE)

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        # Flatten all cells into a single list of strings, skipping completely empty cells
        rows = df.astype(str).apply(lambda x: x.str.strip()).values.flatten().tolist()
        rows = [row for row in rows if row and row.lower() != 'nan']  # Filter empty/nan strings

        i = 0
        while i < len(rows):
            q_candidate = rows[i]
            if question_starters.match(q_candidate.lower()):
                question = q_candidate
                answer = ""
                # Find next non-empty row as answer
                j = i + 1
                while j < len(rows):
                    if rows[j].strip():
                        answer = rows[j].strip()
                        break
                    j += 1
                if question and answer:
                    faqs.append({'category': sheet, 'question': question, 'answer': answer})
                i = j + 1
            else:
                i += 1

    print(f"Extracted {len(faqs)} Q&A pairs from Excel.")
    return faqs


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_text(text: str) -> List[str]:
    doc = nlp.tokenizer(text)
    return [token.text for token in doc]

def preprocess_faqs(faqs: List[Dict]) -> List[Dict]:
    processed = []
    for item in faqs:
        q_clean = clean_text(item['question'])
        a_clean = clean_text(item['answer'])
        combined_text = f"Question: {q_clean}\nAnswer: {a_clean}"
        processed.append({
            'category': item['category'],
            'question': q_clean,
            'answer': a_clean,
            'question_tokens': tokenize_text(q_clean),
            'answer_tokens': tokenize_text(a_clean),
            'text': combined_text  # For FAISS ingestion
        })
    return processed

def save_processed(faqs: List[Dict], filename: str):
    out_path = PROCESSED_DATA_DIR / filename
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(faqs)} records to {out_path}")

def main():
    json_faqs = load_json_faq(RAW_DATA_DIR / 'funds_transfer_app.json')
    excel_faqs = load_excel_qa(RAW_DATA_DIR / 'NUST Bank-Product-Knowledge.xlsx')
    all_faqs = json_faqs + excel_faqs
    processed_faqs = preprocess_faqs(all_faqs)
    save_processed(json_faqs, 'funds_transfer_app_processed.json')
    save_processed(excel_faqs, 'product_knowledge_processed.json')
    save_processed(processed_faqs, 'all_faqs_processed.json')

if __name__ == '__main__':
    main()
