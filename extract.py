import pdfplumber
import spacy
import re
import json
import requests

nlp = spacy.load("fr_core_news_sm")

def remove_numbers(text):
    """
    remove numbers and numeric patterns (like '110,862' or '2 050 947')
    """
    return re.sub(r'\b\d+[\d\s.,-]*\b', '', text).strip()

def extract_word_groups(pdf_path):
    results = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                clean_text = remove_numbers(text)
                doc = nlp(clean_text)
                
                # extract groups that have 3 words or more (to have less group -> faster lmao)
                word_groups = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
                
                if word_groups:
                    results[page_num] = word_groups

    return results

def query_ollama(word_groups):
    ollama_url = "http://localhost:11434/api/generate"

    prompt = (
        "Tu es un assistant intelligent. Voici une liste de groupes de mots extraits d'un document PDF.\n"
        "Retourne un JSON avec chaque groupe de mots associé à une définition claire et concise.\n\n"
        f"{json.dumps(word_groups, indent=2)}\n\n"
        "Format attendu: \n"
        "{ \"definitions\": { \"groupe_de_mot\": \"définition ici\", ... } }"
        "Tu ne répondras que le fichier JSON et absolument rien d'autre !"
    )
    
    print(prompt)

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False  # avoid stream response and get only 1 big response
    }

    response = requests.post(ollama_url, json=payload)
    
    if response.status_code == 200:
        try:
            data = response.json()
            return json.loads(data["response"])
        except json.JSONDecodeError:
            print("failed to decode Ollama response")
            return None
    else:
        print(f"apparently error from Ollama: {response.status_code}, {response.text}")
        return None

pdf_path = "myfile.pdf"
word_groups_per_page = extract_word_groups(pdf_path)

print("\nsending data to Ollama . . .")
definitions = query_ollama(word_groups_per_page)

if definitions:
    final_output = {
        "word_groups": word_groups_per_page,
        "definitions": definitions["definitions"]
    }

    print("\nFinal JSON Output")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\nJSON saved to `output.json`")
else:
    print("No definitions from Ollama")
