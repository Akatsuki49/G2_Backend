from sentence_transformers import SentenceTransformer, util
import redis
from data_cleaner import clean_text_data
from keyword_extraction import keygen
import torch
from rouge_score import rouge_scorer


def get_sent(text):
    # iterate over the text and get sentences: get only those whose len(str.strip()) > 0
    lst = text.split('.')
    final_lst = []
    for i in lst:
        if len(i.strip()) >= 3:
            final_lst.append(i.strip())
    return final_lst
# between cleaned_data and key_phrases


def embedding_similarity_keyphrases(full_context, keyPhrases):

    actual_key_phrases = ""

    for key_phrase in keyPhrases:
        actual_key_phrases = actual_key_phrases + key_phrase + " "

    full_context = full_context['full_text']

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    context_embeddings = model.encode(
        full_context, convert_to_tensor=True)
    keyphrase_embeddings = model.encode(
        actual_key_phrases, convert_to_tensor=True)

    cosine_score = util.pytorch_cos_sim(
        context_embeddings, keyphrase_embeddings)

    return cosine_score.item()


# try doing smtn about this: if this is not necessary try coming up with smtn better
def rouge_score_keyphrases(full_context, keyPhrases):

    full_context = full_context['full_text']
    full_key_phrases = ""

    for key_phrase in keyPhrases:
        full_key_phrases = full_key_phrases + key_phrase + " "

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    rouge_scores = scorer.score(full_key_phrases, full_context)

    rouge1 = rouge_scores['rouge1'].fmeasure
    rougel = rouge_scores['rougeL'].fmeasure

    return rouge1, rougel


def embedding_similarity_summary(summary1, summary2):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    summary1 = summary1.decode("utf-8")

    summary1_embedding = model.encode(summary1, convert_to_tensor=True)
    summary2_embedding = model.encode(summary2, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(
        summary1_embedding, summary2_embedding).item()

    return cosine_scores


def rouge_score_summary(summary1, summary2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary1, summary2)

    summary1 = summary1.decode("utf-8")

    rouge1 = scores['rouge1'].fmeasure
    rougeL = scores['rougeL'].fmeasure

    return rouge1, rougeL


def load_reference_summaries(file_path):
    summaries = []
    with open(file_path, 'r') as f:
        for lines in f:
            lines = lines.strip()
            if (len(lines) > 0):
                summaries.append(lines.strip())

    return summaries


def eval_summary(generated_summary, reference_summaries):
    embedding_similaritiy_sum = 0
    rouge1_score_sum = 0
    rougeL_score_sum = 0

    for ref_summary in reference_summaries:
        similarity = embedding_similarity_summary(
            generated_summary, ref_summary)
        rouge1, rougeL = rouge_score_summary(
            generated_summary, ref_summary)

        embedding_similaritiy_sum += similarity
        rouge1_score_sum += rouge1
        rougeL_score_sum += rougeL

    # Average similarity score
    average_similarity = embedding_similaritiy_sum / \
        len(reference_summaries)
    average_rouge1 = rouge1_score_sum / len(reference_summaries)
    average_rougeL = rougeL_score_sum / len(reference_summaries)

    return average_similarity, average_rouge1, average_rougeL

    # can also have composite score using weighted metrics


if __name__ == "__main__":
    r = redis.Redis(host='shubham', port=6379, db="")

    url = "https://www.chattechnologies.com/"
    data_full_context = r.hgetall(f"scraped:{url}")
    decoded_data = {key.decode('utf-8'): value.decode('utf-8')
                    for key, value in data_full_context.items()}
    full_context = clean_text_data(decoded_data)
    key_phrases = keygen(str(full_context))

    actual_key_phrases = []
    for key_phrase in key_phrases:
        actual_key_phrases.append(key_phrase[0])

    key_phrases = actual_key_phrases

    print("Average KeyPhrase Embedding Similarity: ",
          embedding_similarity_keyphrases(full_context, key_phrases))
    print("Average KeyPhrase Rouge1 and RougeL Scores: ", rouge_score_keyphrases(
        full_context, key_phrases))

    generated_summary = r.get(f"summary:{url}")

    reference_summaries = load_reference_summaries(
        f'C:\\Users\\sowme\\StudioProjects\\G2_Backend\\g2_hack\\summaries.txt')

    # print("Ref : ", reference_summaries, len(reference_summaries))
    # print("Gen : ", generated_summary)
    # generate an evaluation score: average_embedding similarity, average_rouge1, average_rougeL

    average_embedding_similarity, average_rouge1, average_rougeL = eval_summary(
        generated_summary, reference_summaries)

    print("Average Summary Embedding Similarity: ", average_embedding_similarity)
    print("Average Summary Rouge1: ", average_rouge1)
    print("Average Summary RougeL: ", average_rougeL)
    pass
