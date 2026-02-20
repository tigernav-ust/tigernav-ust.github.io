import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd

# Initialize the pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for embeddings to avoid recomputation
embedding_cache = {}

def load_json(file_path):
    """Load JSON file and return data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_embedding(sentence):
    """Get the embedding for a sentence, using cache if available."""
    if not isinstance(sentence, str):
        print(f"Warning: Skipping non-string input: {sentence}")
        return None
    if sentence not in embedding_cache:
        embedding_cache[sentence] = model.encode(sentence)
    return embedding_cache[sentence]

def compute_jaccard_similarity(sentence_1, sentence_2):
    """Compute Jaccard similarity between two sentences."""
    # Tokenize the sentences into sets of words
    set_1 = set(sentence_1.lower().split())
    set_2 = set(sentence_2.lower().split())

    # Compute the intersection and union of the sets
    intersection = len(set_1.intersection(set_2))
    union = len(set_1.union(set_2))

    # Calculate Jaccard similarity
    if union == 0:
        return 0.0  # Prevent division by zero
    return intersection / union

def process_json_file(file_path):
    """Process the JSON file and calculate similarity between chosen and rejected sentences."""
    data = load_json(file_path)
    if not data:
        print(f"No data found in the JSON file {file_path}.")
        return None

    chosen_sentences = [item.get('chosen', '') for item in data if isinstance(item.get('chosen'), str)]
    rejected_sentences = [item.get('rejected', '') for item in data if isinstance(item.get('rejected'), str)]

    if len(chosen_sentences) < 1 or len(rejected_sentences) < 1:
        print(f"Not enough chosen or rejected pairs in the JSON file {file_path}.")
        return None

    similarity_scores = []

    # Compute similarities between chosen and rejected using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(compute_jaccard_similarity, chosen, rejected)
            for chosen, rejected in zip(chosen_sentences, rejected_sentences)
        ]

        for future in futures:
            similarity_scores.append(future.result())

    return similarity_scores, chosen_sentences, rejected_sentences

def calculate_average_similarity(scores):
    """Calculate the average Jaccard similarity score for a list of scores."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def save_summary_to_file(file_name, similarity_scores, output_folder):
    """Save only the overall average score into a text file and full data to Excel."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    overall_avg_similarity = calculate_average_similarity(similarity_scores)

    txt_file = os.path.join(output_folder, f"{file_name}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("Overall Average Jaccard Similarity:\n")
        f.write(f"Chosen vs. Rejected: {overall_avg_similarity:.4f}\n")

    excel_file = os.path.join(output_folder, f"{file_name}.xlsx")
    data = {
        'Chosen vs. Rejected Similarity': similarity_scores
    }
    df = pd.DataFrame(data)
    df.to_excel(excel_file, index=False)
    print(f"Results saved in {txt_file} and {excel_file}")

    return overall_avg_similarity

def process_all_files_in_folder(folder, output_folder):
    """Process all JSON files in the folder."""
    json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    summary_data = []
    total_similarity = 0
    file_count = 0

    for json_file in json_files:
        file_path = os.path.join(folder, json_file)
        result = process_json_file(file_path)

        if result:
            similarity_scores, chosen_sentences, rejected_sentences = result
            file_name = os.path.splitext(json_file)[0]
            overall_avg_similarity = save_summary_to_file(file_name, similarity_scores, output_folder)

            summary_data.append({
                'File Name': file_name,
                'Overall Avg Similarity (Chosen vs. Rejected)': overall_avg_similarity
            })

            total_similarity += overall_avg_similarity
            file_count += 1

    global_avg_similarity = total_similarity / file_count if file_count > 0 else 0

    summary_file = os.path.join(output_folder, 'summary_all_files.xlsx')
    df = pd.DataFrame(summary_data)
    df.loc[len(df)] = {
        'File Name': 'Global Averages',
        'Overall Avg Similarity (Chosen vs. Rejected)': global_avg_similarity
    }

    df.to_excel(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

# Example usage:
if __name__ == "__main__":
    input_folder = "Jaccard Similarity\\DPO"  # Folder containing the JSON files
    output_folder = "Performance Metric (Dataset)\\DPO"  # Folder where output files will be stored
    process_all_files_in_folder(input_folder, output_folder)
