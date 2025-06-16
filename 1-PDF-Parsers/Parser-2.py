import os
import PyPDF2
import networkx as nx
import numpy as np
import pdfplumber
import csv
from PyPDF2 import PdfReader


# Function to extract text from a PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function to extract text from a PDF using pdfplumber (better for structured data)
def extract_text_from_pdf_plumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract additional PDF features (JS, JavaScript, URI, etc.)
def extract_pdf_metadata(pdf_path):
    pdf_metadata = {
        '/JS': False, '/JavaScript': False, '/URI': False, '/Action': False, 
        '/AA': False, '/OpenAction': False, '/launch': False, '/submitForm': False,
        '/Acroform': False, '/XFA': False, '/JBig2Decode': False, '/Colors': False,
        '/Richmedia': False, '/Trailer': False, '/Xref': False, '/Startxref': False
    }
    
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        
        # Extract metadata and check for the specified keys
        try:
            # Extract document metadata
            root = reader.trailer['/Root']
            if '/JS' in root:
                pdf_metadata['/JS'] = True
            if '/JavaScript' in root:
                pdf_metadata['/JavaScript'] = True
            if '/URI' in root:
                pdf_metadata['/URI'] = True
            if '/Action' in root:
                pdf_metadata['/Action'] = True
            if '/AA' in root:
                pdf_metadata['/AA'] = True
            if '/OpenAction' in root:
                pdf_metadata['/OpenAction'] = True
            if '/launch' in root:
                pdf_metadata['/launch'] = True
            if '/submitForm' in root:
                pdf_metadata['/submitForm'] = True
            if '/Acroform' in root:
                pdf_metadata['/Acroform'] = True
            if '/XFA' in root:
                pdf_metadata['/XFA'] = True
            if '/JBig2Decode' in root:
                pdf_metadata['/JBig2Decode'] = True
            if '/Colors' in root:
                pdf_metadata['/Colors'] = True
            if '/Richmedia' in root:
                pdf_metadata['/Richmedia'] = True
            if '/Trailer' in reader.trailer:
                pdf_metadata['/Trailer'] = True
            if '/Xref' in reader.trailer:
                pdf_metadata['/Xref'] = True
            if '/Startxref' in reader.trailer:
                pdf_metadata['/Startxref'] = True
        except Exception as e:
            print(f"Error extracting metadata for {pdf_path}: {e}")

    return pdf_metadata

# Function to parse graph data from the extracted text
def parse_graph_from_text(text):
    G = nx.Graph()
    lines = text.splitlines()
    for line in lines:
        if len(line.split()) == 2:
            node1, node2 = line.split()
            G.add_edge(node1, node2)
    return G

# Function to compute graph features
def compute_graph_features(G):
    features = {}
    
    degrees = dict(G.degree())
    features['avg_degree'] = np.mean(list(degrees.values()))
    features['avg_clustering_coefficient'] = nx.average_clustering(G)
    
    try:
        features['avg_shortest_path'] = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        features['avg_shortest_path'] = float('inf')
    
    features['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    features['density'] = nx.density(G)
    features['median_children'] = np.median(list(degrees.values()))
    features['num_edges'] = G.number_of_edges()
    features['num_leaves'] = sum(1 for degree in degrees.values() if degree == 1)
    features['num_nodes'] = G.number_of_nodes()
    features['var_children'] = np.var(list(degrees.values()))
    
    return features

def collect_pdf_info(pdf_path):
    info = {}

    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        
        # PDF File Size
        info['PDF_size'] = os.path.getsize(pdf_path)
        
        # Title characters
        info['title_characters'] = len(reader.metadata.get('/Title', ''))
        
        # Encryption status
        info['encryption'] = reader.is_encrypted
        
        # Metadata size
        metadata = reader.metadata
        info['metadata_size'] = sum(len(value) for key, value in metadata.items() if value)
        
        # Number of pages
        info['page_number'] = len(reader.pages)
        
        # Extract information on the streams, fonts, and other objects
        header = len(reader.trailer)
        info['header'] = header
        
        # Handle image counting (if any)
        image_count = 0
        for page in reader.pages:
            if '/Resources' in page and '/XObject' in page['/Resources']:
                image_count += len(page['/Resources']['/XObject'])
        info['image_number'] = image_count
        
        # Text content length (just counting characters of extracted text)
        text_count = 0
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_count += len(text)
        info['text'] = text_count
        
        # Object numbers (based on /Pages -> /Kids)
        # Dereference the indirect objects
        try:
            root = reader.trailer['/Root']
            pages = root['/Pages']
            kids = pages['/Kids']
            info['object_number'] = len(kids) if isinstance(kids, list) else 0
        except KeyError:
            info['object_number'] = 0
        
        # Embedded files (if any)
        embedded_files = reader.trailer['/Root'].get('/Names', {}).get('/EmbeddedFiles', {}).get('/Names', [])
        info['number_of_embedded_files'] = len(embedded_files) // 2 if embedded_files else 0
        
        # Average size of all embedded media (if any)
        if embedded_files:
            total_size = sum(os.path.getsize(f) for f in embedded_files)
            info['average_size_of_all_embedded_media'] = total_size / len(embedded_files) if embedded_files else 0
        
        # Number of streams
        info['number_of_streams'] = len(reader.pages)
        
        # Average stream size (size of /Contents in each page)
        stream_sizes = []
        for page in reader.pages:
            contents = page.get('/Contents', [])
            if isinstance(contents, list):
                for content in contents:
                    stream = content.get_object()
                    stream_sizes.append(len(stream.get_data()) if hasattr(stream, 'get_data') else 0)
            elif isinstance(contents, PyPDF2.generic.IndirectObject):
                stream = contents.get_object()
                stream_sizes.append(len(stream.get_data()) if hasattr(stream, 'get_data') else 0)
        
        info['average_stream_size'] = np.mean(stream_sizes) if stream_sizes else 0
        
        # Number of nested filters (this would need parsing of /Filter in streams; we'll assume 0 here)
        info['number_of_nested_filters'] = 0  # Complex parsing required for nested filters
        
        # Number of Xref entries
        xref_entries = reader.trailer.get('/Xref', None)
        info['number_of_Xref_entries'] = len(xref_entries) if xref_entries else 0
        
        # Number of name obfuscations (complex analysis, will not be implemented directly)
        info['number_of_name_obfuscations'] = 0  # Requires more complex PDF parsing
        
        # Total number of filters used (filter chains in streams)
        info['total_number_of_filters_used'] = 0  # Requires parsing /Filter in /Streams
        
    return info

# Function to write features to CSV (with file name)
def write_features_to_csv(features, file_name, csv_file):
    # Add the file name as the first column
    features['file_name'] = file_name
    
    fieldnames = ['file_name'] + list(features.keys())
    
    # Check if the CSV file exists
    try:
        with open(csv_file, mode='x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()  # Write the header row
            writer.writerow(features)  # Write the features
    except FileExistsError:
        # If file already exists, append to it
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(features)  # Write the features

# Function to process all PDF files in a folder and store their features in a CSV file
def process_pdfs_in_folder(folder_path, csv_file):
    # Iterate through all PDF files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):  # Check if the file is a PDF
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            
            #  To Extract text from PDF
            text = extract_text_from_pdf_plumber(pdf_path)
            
            #  To Parse graph structure from the text (assuming an edge list format)
            G = parse_graph_from_text(text)
            
            # To Compute graph features
            graph_features = compute_graph_features(G)
            
            # To Collect PDF information (size, encryption, title, etc.)
            pdf_info = collect_pdf_info(pdf_path)
            
            # To Extract additional PDF metadata (JS, JavaScript, URI, etc.)
            pdf_metadata = extract_pdf_metadata(pdf_path)
            
            # To Combine all features (graph, PDF info, metadata)
            combined_features = {**graph_features, **pdf_info, **pdf_metadata}
            
            # To Write features to CSV (including file name)
            write_features_to_csv(combined_features, filename, csv_file)

# Example usage
folder_path = "/home/sharmila/MyCodes/PDFMalware"  # Change this to your folder path
csv_file = "pdf_features2.csv"   # This contains graph based features for PDF
process_pdfs_in_folder(folder_path, csv_file)

print(f"Features from all PDFs in {folder_path} have been written to {csv_file}")
