import streamlit as st


def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    words = sorted(set([line.strip() for line in lines]))
    return words


def calculate_levenshtein_distance(s1, s2):
    # Step 1: Initialize matrix D
    m = len(s1) + 1
    n = len(s2) + 1
    D = [[0 for _ in range(n)] for _ in range(m)]
    # Step 2: Initialize first row and first column
    D[0] = [i for i in range(n)]
    for i in range(m):
        D[i][0] = i
    # Step 3: Fill the whole matrix
    for i in range(1, m):
        for j in range(1, n):
            if s1[i-1] == s2[j-1]:
                D[i][j] = D[i-1][j-1]
            else:
                # Insert, Delete, Substitute accordingly
                D[i][j] = min(D[i-1][j] + 1,
                              D[i][j-1] + 1,
                              D[i-1][j-1] + 1)
    # Step 4: Return the last element which is the distance
    return D[-1][-1]


def main():
    vocabs = load_vocab(file_path='./data/vocab.txt')
    st.title("Word Correction using Levenshtein Distance")
    word = st.text_input('Word:')
    if st.button("Compute"):
        # compute levenshtein distance
        leven_distances = dict()
        for vocab in vocabs:
            leven_distances[vocab] = calculate_levenshtein_distance(
                word, vocab)

        # sorted by distance
        sorted_distances = dict(
            sorted(leven_distances.items(), key=lambda item: item[1]))
        correct_word = list(sorted_distances.keys())[0]
        st.write('Correct word: ', correct_word)

        col1, col2 = st.columns(2)
        col1.write('Vocabulary:')
        col1.write(vocabs)

        col2.write('Levenshtein Distance:')
        col2.write(sorted_distances)


if __name__ == '__main__':
    main()
