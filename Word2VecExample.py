import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


f = open('hurriyet.txt', 'r', encoding='utf8')
#türkçe karakterlerle çalıştığımız için encoding='utf8' eklendi

text = f.read()
#text değişkeninde artık her cümle yeni bir satırla ayrıldı

#datasetteki tüm noktalama işaretleri silindiği için ve tüm harfler küçük harf olduğu için .split yeterli olacak

t_list = text.split('\n')

corpus = []

for cumle in t_list:
    corpus.append(cumle.split())

print(corpus[:10])

#corpusu, kelime vektör uzunluğunu ve window size alır(ortadaki kelimenin sağından ve solundan 5 adet kelime dikkate alınır)
#min_count = 5 ,corpus içerisinde en az 5 kere geçen kelimeleri dikkate al
#sg = 1, skip-gram kullanılacağını belirtiyoruz, default=CBOW
model = Word2Vec(corpus, size=100, window=5, min_count=5, sg=1)

#kelime vektörlerini incelemek:
ankara = model.wv['ankara']
print(ankara)

#hollanda kelimesine yakın kelimeleri görmek:
similar = model.wv.most_similar('hollanda')
print(similar)

#modeli daha sonra kullanmak için kaydettik
#model.save('word2vec.model')

#model = Word2Vec.load('word2vec.model')

def closestwords_tsneplot(model,word):
    word_vectors = np.empty((0,100))
    word_labels = [word]

    close_words = model.wv.most_similar(word)

    #kelimelerin vektörleri dizi içerisinde toplandı
    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)

    for w, _ in close_words:
        word_labels.append(w)
        word_vectors = np.append(word_vectors, np.array([model.wv[w]]), axis=0)

    # TSNE vektörleri grafikleştirmemizi sağlayacak olan bir ML algoritması
    tsne = TSNE(random_state=0)
    Y = tsne.fit_transform(word_vectors)

    #koordinatlar ayrı ayrı alındı
    x_coord = Y[:,0]
    y_coord = Y[:,1]


    plt.scatter(x_coord, y_coord)

    for label, x, y in zip(word_labels, x_coord,y_coord):
        plt.annotate(label, xy=(x,y), xytext=(5,-2), textcoords='offset points')

    plt.show()

graph = closestwords_tsneplot(model, 'temmuz')
print(graph)
