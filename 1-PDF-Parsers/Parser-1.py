import os
import PyPDF2
import networkx as nx
import numpy as np
import pdfplumber
import csv
import fitz  # PyMuPDF
from datetime import datetime
import re
from PyPDF2 import PdfReader
from PyPDF2.generic import IndirectObject

def extract_text_from_pdf_plumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

def extract_pdf_metadata(pdf_path):
    pdf_metadata = {'/JS': False, '/JavaScript': False, '/URI': False, '/Action': False, 
                    '/AA': False, '/OpenAction': False, '/launch': False, '/submitForm': False,
                    '/Acroform': False, '/XFA': False, '/JBig2Decode': False, '/Colors': False,
                    '/Richmedia': False, '/Trailer': False, '/Xref': False, '/Startxref': False}
    
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        try:
            root = reader.trailer.get('/Root', {})
            if isinstance(root, IndirectObject):
                root = root.get_object()
            for key in pdf_metadata.keys():
                pdf_metadata[key] = key in root if isinstance(root, dict) else False
        except Exception as e:
            print(f"Error extracting metadata for {pdf_path}: {e}")
    return pdf_metadata

def parse_graph_from_text(text):
    G = nx.Graph()
    lines = text.splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            G.add_edge(parts[0], parts[1])
    return G

def compute_graph_features(G):
    features = {}
    degrees = list(dict(G.degree()).values())
    features['num_nodes'] = G.number_of_nodes()
    features['num_edges'] = G.number_of_edges()
    features['avg_degree'] = np.mean(degrees) if degrees else 0
    features['density'] = nx.density(G)
    features['avg_clustering_coefficient'] = nx.average_clustering(G) if features['num_nodes'] > 1 else 0
    return features

def count_character_types(text):
    if text:
        return {
            'dot': text.count('.'),
            'len': len(text),
            'num': sum(c.isdigit() for c in text),
            'oth': len(re.sub(r'[a-zA-Z0-9]', '', text)),
            'uc': sum(c.isupper() for c in text)
        }
    return {'dot': 0, 'len': 0, 'num': 0, 'oth': 0, 'uc': 0}

def extract_pdf_features(pdf_path):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata or {}
    features = {'count_page': doc.page_count, 'size': os.path.getsize(pdf_path)}
    
    for field in ['author', 'creator', 'producer', 'keywords', 'subject', 'title']:
        text = metadata.get(field, "")
        features.update({f"{field}_{key}": value for key, value in count_character_types(text).items()})
    
    features['createdate_ts'] = metadata.get("creationDate", "unknown")
    features['moddate_ts'] = metadata.get("modDate", "unknown")
    features['count_font'] = sum(page.get_text("raw").count("/Font") for page in doc)

    return features

def write_features_to_csv(features, file_name, csv_file):
    features['file_name'] = file_name

    # Convert missing values to appropriate types
    for key, value in features.items():
        if value is None:
            if isinstance(value, int) or key.startswith("count_") or key.startswith("num_"):
                features[key] = 0  # Default for missing numeric values
            else:
                features[key] = "unknown"  # Default for missing string values

    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=features.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(features)

def process_pdfs_in_folder(folder_path, csv_file):
    for root, _, files in os.walk(folder_path):  # Recursively iterate over subdirectories
        for filename in files:
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, filename)
                print(f"Processing {pdf_path}...")
                text = extract_text_from_pdf_plumber(pdf_path)
                G = parse_graph_from_text(text)
                graph_features = compute_graph_features(G)
                pdf_metadata = extract_pdf_metadata(pdf_path)
                pdf_info = extract_pdf_features(pdf_path)
                combined_features = {**graph_features, **pdf_metadata, **pdf_info}
                write_features_to_csv(combined_features, filename, csv_file)

# Example usage
folder_path = "/home/sharmila/MyCodes/PDFMalware"
csv_file = "pdf_features1.csv"
process_pdfs_in_folder(folder_path, csv_file)
print(f"Features from all PDFs in {folder_path} have been written to {csv_file}")
