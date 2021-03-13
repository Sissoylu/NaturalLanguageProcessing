import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, CuDNNGRU
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('hepsiburada.csv')

#verileri ve etiketleri alıp liste haline getirdik.
target = dataset['Rating'].values.tolist()
data = dataset['Review'].values.tolist()

#verileri eğitim ve test için ayırmalıyız.(verileri parçalayacağız)

#verilerin %80i ne kadar?
cutoff = int(len(data)*0.80)

#oluşturulan ayırma noktasından verileri eğitim ve test için ayırdık
x_train, x_test = data[:cutoff], data[cutoff:]

#etiketleri de ayırdık
y_train, y_test = target[:cutoff], target[cutoff:]

#print(x_train[500])

#olumlu yorumların etiketi=1, olumsuz yorumların etiketi=0


#en sık geçen 10000 kelime alınacak, diğerleri atılacak
num_words = 10000

tokenizer = Tokenizer(num_words=num_words)

#tüm yorumlar tokenleştiriliyor.
tokenizer.fit_on_texts(data)

#kelimelerin kullanılma sayısını gösterir
wordnum = tokenizer.word_index
#print(wordnum)

#string halindeki kelimeleri tokenleştirmeliyiz, üzerinde işlem yapabilmek için
x_train_tokens = tokenizer.texts_to_sequences(x_train)

#yazılan cümle artık sayılarla ifade ediliyor.
#print(x_train[800])
#print(x_train_tokens[800])

#test kümesi tokenleştiriliyor
x_test_tokens = tokenizer.texts_to_sequences(x_test)

#boyut belirlenip eksik boyutta kalırsa 0 eklenerek tamamlanır, yüksek boyutta kalırsa ekstralar silinir

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

#yorumlarda ortalama kaç token var?
ortalama = np.mean(num_tokens)
#print(ortalama)

enCok = np.max(num_tokens)
#print(enCok)

#en uzun yorum
index = np.argmax(num_tokens)
#print(x_train[21941])


max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
#print(max_tokens)
#output = 59, yani 59 tokenli olacak. boyutlar 59a setlenecek

#np.sum(num_tokens<max_tokens)/len(num_tokens)

#padding eklenecek, eğitim setindeki her yorum eşit seviyeye getirildi (59 token)
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)

x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

shape = x_train_pad.shape
#print(shape) 190binden fazla yorum 59 uzunluğunda artık


#print(np.array(x_train_tokens[800]))
#print("\n")
#print(x_train_pad[800])

#kelimelerin sayısal tokenleri bulunuyor
idx = tokenizer.word_index

#word_index'i tersine çevirdik.
#bir sayıyı verdiğimiz zaman o sayıya karşılık gelen kelimeyi verir
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token!= 0]
    text = ' '.join(words)
    return text

yorum = x_train[800]
yorum2 = tokens_to_string(x_train_tokens[800])

#aynı yorum elde edildi, fonksiyon doğru çalışıyor.
print(yorum)
print(yorum2)
#10000 kelime içerisinde olmayan kelimeler çıkarıldı


#embedding
model = Sequential()

embedding_size = 50 #her kelimeye karşılık gelen 50 uzunluğunda bir vektör

#10bin kelime 50 uzunluğunda
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer'))

#units=nöron sayısı
model.add(CuDNNGRU(units=16, return_sequences=True))
model.add(CuDNNGRU(units=8, return_sequences=True))
model.add(CuDNNGRU(units=4))
#sigmoid ile sinir ağı 0-1 aralığına alındı
#sonuç 1e yakınsa olumlu, diğer durumda olumsuz
model.add(Dense(1, activation='sigmoid'))

optimizer= Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#sum = model.summary()
#print(sum)

#eğitim setindeki tüm verilerin bir kere eğitimden geçmesine epoch denir
#veriler 5 kere eğitilecek
model.fit(x_train_pad, y_train, epochs=5, batch_size=256)

result = model.evaluate(x_test_pad, y_test)

print(result[1])












