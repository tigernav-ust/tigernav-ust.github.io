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

def compute_similarity(sentence_1, sentence_2):
    """Compute cosine similarity between two sentences."""
    embedding_1 = get_embedding(sentence_1)
    embedding_2 = get_embedding(sentence_2)
    
    # Skip if embeddings are None
    if embedding_1 is None or embedding_2 is None:
        return 0.0
    
    similarity = cosine_similarity([embedding_1], [embedding_2])
    return similarity[0][0]

def calculate_average_similarity(scores):
    """Calculate the average cosine similarity score for a list of scores."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def process_json_file(file_path):
    """Process the JSON file and calculate the similarity."""
    data = load_json(file_path)
    if not data:
        print(f"No data found in the JSON file {file_path}.")
        return None

    questions = [item.get('Question', '') for item in data if isinstance(item.get('Question'), str)]
    answers = [item.get('Answer', '') for item in data if isinstance(item.get('Answer'), str)]

    if len(questions) < 2 or len(answers) < 2:
        print(f"Not enough Q&A pairs in the JSON file {file_path}.")
        return None

    # Process the data asynchronously in batches
    batch_size = 10
    avg_question_scores = []
    avg_answer_scores = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            futures.append(executor.submit(compute_batch_similarity, batch_questions))

        for future in futures:
            avg_question_scores.extend(future.result())

        futures.clear()  # Clear futures to reuse for answers

        for i in range(0, len(answers), batch_size):
            batch_answers = answers[i:i + batch_size]
            futures.append(executor.submit(compute_batch_similarity, batch_answers))

        for future in futures:
            avg_answer_scores.extend(future.result())

    return avg_question_scores, avg_answer_scores, questions, answers

def compute_batch_similarity(sentences):
    """Compute similarity for a batch of sentences."""
    scores = []
    for i, sent1 in enumerate(sentences):
        if not isinstance(sent1, str):
            continue
        batch_scores = []
        for j, sent2 in enumerate(sentences):
            if i != j and isinstance(sent2, str):
                similarity = compute_similarity(sent1, sent2)
                batch_scores.append(similarity)
        if batch_scores:
            avg_score = calculate_average_similarity(batch_scores)
            scores.append(avg_score)
    return scores

def save_summary_to_file(file_name, avg_question_scores, avg_answer_scores, output_folder):
    """Save only the overall average scores into a text file and full data to Excel."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    overall_avg_question = calculate_average_similarity(avg_question_scores)
    overall_avg_answer = calculate_average_similarity(avg_answer_scores)

    txt_file = os.path.join(output_folder, f"{file_name}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("Overall Average Cosine Similarity:\n")
        f.write(f"Questions: {overall_avg_question:.4f}\n")
        f.write(f"Answers: {overall_avg_answer:.4f}\n")

    max_length = max(len(avg_question_scores), len(avg_answer_scores))
    avg_question_scores += [None] * (max_length - len(avg_question_scores))
    avg_answer_scores += [None] * (max_length - len(avg_answer_scores))

    excel_file = os.path.join(output_folder, f"{file_name}.xlsx")
    data = {
        'Avg Question Similarity': avg_question_scores,
        'Avg Answer Similarity': avg_answer_scores
    }
    df = pd.DataFrame(data)
    df.to_excel(excel_file, index=False)
    print(f"Results saved in {txt_file} and {excel_file}")

    return overall_avg_question, overall_avg_answer

def save_summary_for_all_files(summary_data, output_folder, global_avg_question, global_avg_answer):
    """Save the summary of all files into a single Excel file and global averages."""
    summary_file = os.path.join(output_folder, 'summary_all_files.xlsx')
    df = pd.DataFrame(summary_data)

    df.loc[len(df)] = {
        'File Name': 'Global Averages',
        'Overall Avg Question Similarity': global_avg_question,
        'Overall Avg Answer Similarity': global_avg_answer
    }

    df.to_excel(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

def process_all_files_in_folder(folder, output_folder):
    """Process all JSON files in the folder."""
    json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    summary_data = []
    total_question_similarity = 0
    total_answer_similarity = 0
    file_count = 0

    for json_file in json_files:
        file_path = os.path.join(folder, json_file)
        result = process_json_file(file_path)

        if result:
            avg_question_scores, avg_answer_scores, questions, answers = result
            file_name = os.path.splitext(json_file)[0]
            overall_avg_question, overall_avg_answer = save_summary_to_file(file_name, avg_question_scores, avg_answer_scores, output_folder)

            summary_data.append({
                'File Name': file_name,
                'Overall Avg Question Similarity': overall_avg_question,
                'Overall Avg Answer Similarity': overall_avg_answer
            })

            total_question_similarity += overall_avg_question
            total_answer_similarity += overall_avg_answer
            file_count += 1

    global_avg_question = total_question_similarity / file_count if file_count > 0 else 0
    global_avg_answer = total_answer_similarity / file_count if file_count > 0 else 0

    save_summary_for_all_files(summary_data, output_folder, global_avg_question, global_avg_answer)

# Example usage:
if __name__ == "__main__":
    input_folder = "Cosine Similarity\SFT Dataset"  # Folder containing the JSON files
    output_folder = "Performance Metric (Dataset)\SFT"  # Folder where output files will be stored
    process_all_files_in_folder(input_folder, output_folder)
