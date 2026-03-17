import math
import time
import random
from collections import Counter
import numpy as np
import pulp

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ===============================
# HYPERPARAMETERS
# ===============================
TOP_K = 3
POWER_ITER = 50
DAMPING = 0.85
EPSILON = 1e-6

SENT_DROP_RATE = 0.25
NUM_SENT_DROPOUTS = 4

DROP_EDGE_RATE = 0.5
NUM_EDGE_DROPOUTS = 10

LEAD_BIAS = 0.30
LAMBDA_REDUNDANCY = 0.3
MMR_LAMBDA = 0.7


# ===============================
# NLP PREP
# ===============================
def initialize_nlp():
    return set(stopwords.words("english")), WordNetLemmatizer()


def get_sentences(text):
    return sent_tokenize(text)


def tokenize_sentences(sentences, stop_words, lemmatizer):
    return [
        [lemmatizer.lemmatize(w.lower())
         for w in word_tokenize(s)
         if w.isalpha() and w.lower() not in stop_words]
        for s in sentences
    ]


# ===============================
# TOKEN IMPORTANCE + ISF
# ===============================
def compute_token_importance(tokenized_sents):
    tf = Counter()
    for s in tokenized_sents:
        tf.update(s)
    return {w: 1 + math.log(1 + c) for w, c in tf.items()}


def compute_isf(tokenized_sents):
    N = len(tokenized_sents)
    sf = Counter()
    for s in tokenized_sents:
        for w in set(s):
            sf[w] += 1
    return {w: math.log((N + 1) / (sf[w] + 1)) for w in sf}


def sentence_vector(tokens, token_imp, isf):
    vec = {}
    L = len(tokens)
    if L == 0:
        return vec

    for w in tokens:
        vec[w] = vec.get(w, 0.0) + token_imp.get(w, 1.0) * isf.get(w, 0.0)

    for w in vec:
        vec[w] /= L

    return vec


def cosine_similarity(v1, v2):
    words = set(v1) | set(v2)
    num = sum(v1.get(w, 0) * v2.get(w, 0) for w in words)

    d1 = math.sqrt(sum(v * v for v in v1.values()))
    d2 = math.sqrt(sum(v * v for v in v2.values()))

    if d1 < 1e-12 or d2 < 1e-12:
        return 0.0

    return num / (d1 * d2)


# ===============================
# BUILD SIMILARITY MATRIX
# ===============================
def build_similarity_matrix(tokens):
    n = len(tokens)
    token_imp = compute_token_importance(tokens)
    isf = compute_isf(tokens)
    vectors = [sentence_vector(s, token_imp, isf) for s in tokens]

    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i][j] = cosine_similarity(vectors[i], vectors[j])
    return M


# ===============================
# SPARSE SENTENCE REMOVAL
# ===============================
def remove_sparse_sentences(matrix, zero_threshold=0.001, sparse_threshold=0.8):
    n = matrix.shape[0]

    row_zero_ratio = np.sum(matrix < zero_threshold, axis=1) / n
    col_zero_ratio = np.sum(matrix < zero_threshold, axis=0) / n

    remove = set(np.where(row_zero_ratio > sparse_threshold)[0]) | \
             set(np.where(col_zero_ratio > sparse_threshold)[0])

    keep = [i for i in range(n) if i not in remove]

    if not keep:
        keep = list(range(n))

    return keep


# ===============================
# SENTENCE DROPOUT
# ===============================
def sentence_dropout_ensemble(tokens):
    n = len(tokens)
    acc = np.zeros((n, n))
    counts = np.zeros((n, n))

    drop_size = max(1, int(SENT_DROP_RATE * n))

    for _ in range(NUM_SENT_DROPOUTS):
        drop = set(random.sample(range(n), min(drop_size, n)))
        keep = [i for i in range(n) if i not in drop]

        if len(keep) == 0:
            continue

        sub_tokens = [tokens[i] for i in keep]

        token_imp = compute_token_importance(sub_tokens)
        isf = compute_isf(sub_tokens)
        vectors = [sentence_vector(s, token_imp, isf) for s in sub_tokens]

        for a, i in enumerate(keep):
            for b, j in enumerate(keep):
                if i != j:
                    sim = cosine_similarity(vectors[a], vectors[b])
                    acc[i][j] += sim
                    counts[i][j] += 1

    result = np.zeros_like(acc)
    np.divide(acc, counts, out=result, where=counts > 0)

    return result


# ===============================
# EDGE DROPOUT
# ===============================
def edge_dropout(M):
    mask = np.random.rand(*M.shape) > DROP_EDGE_RATE
    np.fill_diagonal(mask, 0)
    return M * mask


def edge_dropout_ensemble(M):
    acc = np.zeros_like(M)
    for _ in range(NUM_EDGE_DROPOUTS):
        acc += edge_dropout(M)
    return acc / NUM_EDGE_DROPOUTS


# ===============================
# GRAPH STABILIZATION
# ===============================
def stabilize_graph(M):
    n = len(M)
    return M + EPSILON * np.ones((n, n))


def normalize_matrix(M):
    n = len(M)
    for i in range(n):
        s = M[i].sum()
        if s > 1e-12:
            M[i] = M[i] / s
        else:
            M[i] = np.ones(n) / n
    return M


# ===============================
# PAGERANK
# ===============================
def power_method(M):
    n = len(M)
    P = np.ones(n) / n
    teleport = np.ones(n) / n

    for _ in range(POWER_ITER):
        P = DAMPING * M.T.dot(P) + (1 - DAMPING) * teleport

    return P


# ===============================
# SAFE ILP
# ===============================
def ilp_selection(scores, similarity_matrix):
    n = len(scores)
    similarity_matrix = np.nan_to_num(similarity_matrix)

    prob = pulp.LpProblem("SummarySelection", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    y = {}

    for i in range(n):
        for j in range(i + 1, n):
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    prob += (
        pulp.lpSum(scores[i] * x[i] for i in range(n))
        - LAMBDA_REDUNDANCY * pulp.lpSum(
            similarity_matrix[i][j] * y[(i, j)]
            for i in range(n)
            for j in range(i + 1, n)
        )
    )

    prob += pulp.lpSum(x[i] for i in range(n)) <= min(TOP_K, n)

    for i in range(n):
        for j in range(i + 1, n):
            prob += y[(i, j)] <= x[i]
            prob += y[(i, j)] <= x[j]
            prob += y[(i, j)] >= x[i] + x[j] - 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = []
    if pulp.LpStatus[status] == "Optimal":
        for i in range(n):
            val = pulp.value(x[i])
            if val is not None and val > 0.5:
                selected.append(i)

    if len(selected) == 0:
        selected = list(np.argsort(scores)[-min(TOP_K, n):])

    return selected


# ===============================
# SAFE MMR
# ===============================
def mmr_refinement(selected_idx, scores, similarity_matrix):
    if not selected_idx:
        return list(np.argsort(scores)[-min(TOP_K, len(scores)):])

    selected = []
    candidates = selected_idx.copy()

    while len(selected) < min(TOP_K, len(candidates)):
        best = None
        best_score = -1e9

        for i in candidates:
            if i in selected:
                continue

            redundancy = 0
            if selected:
                redundancy = max(similarity_matrix[i][j] for j in selected)

            mmr_score = MMR_LAMBDA * scores[i] - (1 - MMR_LAMBDA) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best = i

        if best is None:
            break

        selected.append(best)

    return sorted(selected)


# ===============================
# MAIN
# ===============================
def sigmabot(text):
    start = time.time()

    stop_words, lemmatizer = initialize_nlp()
    sentences = get_sentences(text)

    if len(sentences) <= TOP_K:
        return text

    tokens = tokenize_sentences(sentences, stop_words, lemmatizer)

    # Build initial graph
    base_M = build_similarity_matrix(tokens)

    # Sparse removal
    keep = remove_sparse_sentences(base_M)
    sentences = [sentences[i] for i in keep]
    tokens = [tokens[i] for i in keep]

    # Rebuild graph after filtering
    M = sentence_dropout_ensemble(tokens)

    # Edge dropout
    M = edge_dropout_ensemble(M)

    # Stabilize + normalize
    M = stabilize_graph(M)
    M = normalize_matrix(M)

    # PageRank
    scores = power_method(M)

    # Lead bias
    for i in range(len(scores)):
        scores[i] += LEAD_BIAS * (1 - i / len(scores))

    # ILP + MMR
    selected_idx = ilp_selection(scores, M)
    selected_idx = mmr_refinement(selected_idx, scores, M)

    summary = " ".join(sentences[i] for i in selected_idx)

    print("Execution Time:", round(time.time() - start, 2), "sec")
    return summary
