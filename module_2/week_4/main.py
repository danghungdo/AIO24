import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. Basic Probability
def compute_mean(arr):
    sum = 0
    for i in arr:
        sum += i
    return sum / len(arr)


def compute_median(arr):
    size = len(arr)
    arr = np.sort(arr)
    if size % 2 == 0:
        return (arr[size // 2] + arr[size // 2 - 1]) / 2
    else:
        return arr[(size + 1) // 2]


def compute_std(arr):
    mean = compute_mean(arr)
    variance = 0
    for i in arr:
        variance += (i - mean) ** 2
    variance = variance / len(arr)
    return np.sqrt(variance)


def compute_correlation_coefficient(arr1, arr2):
    numerator = 0
    denominator = 0
    mean1 = compute_mean(arr1)
    mean2 = compute_mean(arr2)
    std1 = compute_std(arr1)
    std2 = compute_std(arr2)
    for i in range(len(arr1)):
        numerator += (arr1[i] - mean1) * (arr2[i] - mean2)
    denominator = len(arr1) * std1 * std2
    return np.round(numerator / denominator, 2)

# 3. Text Retrieval


def tfidf_search(question, context_embedded, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question])
    cosine_scores = cosine_similarity(
        query_embedded, context_embedded).flatten()
    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            "id": idx,
            "cosine_score": cosine_scores[idx]
        }
        results.append(doc_score)
    return results


def corr_search(question, context_embedded, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question])
    corr_scores = np.corrcoef(query_embedded.toarray()[
                              0], context_embedded.toarray())
    corr_scores = corr_scores[0][1:]
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            "id": idx,
            "corr_score": corr_scores[idx]
        }
        results.append(doc_score)
    return results


if __name__ == "__main__":
    # Question 1
    X1 = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
    print("Mean: ", compute_mean(X1))
    print("--------------------")
    # Question 2
    X2 = [1, 5, 4, 4, 9, 13]
    print("Median: ", compute_median(X2))
    print("--------------------")
    # Question 3
    X3 = [171, 176, 155, 167, 169, 182]
    print("Standard Deviation: ", compute_std(X3))
    print("--------------------")
    # Question 4
    X4 = np.asarray([-2, -5, -11, 6, 4, 15, 9])
    Y4 = np.asarray([4, 25, 121, 36, 16, 225, 81])
    print("Correlation Coefficient: ", compute_correlation_coefficient(X4, Y4))
    print("--------------------")
    # Question 5
    data = pd.read_csv("advertising.csv")
    X5 = data["TV"]
    Y5 = data["Radio"]
    corr_X5_Y5 = compute_correlation_coefficient(X5, Y5)
    print("Correlation between TV and Radio: ", round(corr_X5_Y5, 2))
    print("--------------------")
    # Question 6
    features = ["TV", "Radio", "Newspaper"]
    for feature_1 in features:
        for feature_2 in features:
            correlation_value = compute_correlation_coefficient(
                data[feature_1], data[feature_2])
            print("Correlation between", feature_1, "and",
                  feature_2, "is", round(correlation_value, 2))
    print("--------------------")
    # Question 7
    X7 = data["Radio"]
    Y7 = data["Newspaper"]
    print("Correlation between Radio and Newspaper:\n", np.corrcoef(X7, Y7))
    print("--------------------")
    # Question 8
    print(data.corr())
    print("--------------------")
    # Question 9
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", linewidths=.5)
    plt.show()
    print("--------------------")
    # Question 10
    vi_data_df = pd.read_csv("vi_text_retrieval.csv")
    context = vi_data_df["text"]
    context = [doc.lower() for doc in context]
    tfidf_vectorizer = TfidfVectorizer()
    context_embedded = tfidf_vectorizer.fit_transform(context)
    print(context_embedded.toarray()[7][0])
    # Question 11
    question = vi_data_df.iloc[0]["question"]
    results = tfidf_search(question, context_embedded, tfidf_vectorizer)
    print(results[0]["cosine_score"])
    print("--------------------")
    # Question 12
    results = corr_search(question, context_embedded, tfidf_vectorizer)
    print(results[1]["corr_score"])
    print("--------------------")
