from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def multiSummarizer(filenames, n):
    docs = []
    for filen in filenames:
        with open('text_file/'+filen, "r") as fd:
            multifile_text = fd.readlines()
            header =[]
            para = []
            for line in multifile_text:
                if len(line) > 1:
                    if len(line) < 100:
                        header.append(line)
                    else:
                        para.append(line)  
            multi_fulltext = ''.join( lines for lines in para)
            print(len(multi_fulltext),'.......................')
            docs.append(multi_fulltext)
    
    print('docs=',len(docs),'.......................')
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(docs)
    similarity_matrix = cosine_similarity(X)
    scores = np.ones(len(docs))
    damping_factor = 0.85
    tol = 1e-5
    max_iter = 100
    for i in range(max_iter):
        prev_scores = scores.copy()
        for j in range(len(docs)):
            scores[j] = (1 - damping_factor) + damping_factor * \
                        np.sum(similarity_matrix[j] * scores / np.sum(similarity_matrix[j]))
        if np.linalg.norm(scores - prev_scores) < tol:
            break
                    
    sentence_scores = list(zip(range(len(docs)), scores))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [x[0] for x in sentence_scores[:n]]
    selected_indices.sort()
    summary = ' '.join([nltk.sent_tokenize(docs[i])[0] for i in selected_indices])
    print('summary=',summary,'..........................')
    return summary