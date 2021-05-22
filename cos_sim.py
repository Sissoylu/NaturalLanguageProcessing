from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def benzerbul(something):

    doc_exist = "yarın bilet"
    doc_input = something

    documents = [doc_input, doc_exist]

    #create the document term matrix
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)


    #convert sparse matrix to pandas dataframe if you want to see the word frequency

    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names(),index=['doc_input','doc_exist'])


    cos_sim = cosine_similarity(df,df)
    print(cos_sim)
    print("\n")

    if cos_sim[0, 1] >= 0.5:
        a = "başarılı"
    else:
        a = "başarısız"

    return a



d_input = input("Bir şeyler")
func_outp= benzerbul(d_input)

print(func_outp)
