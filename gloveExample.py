from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors #modeli yüklemek için

glove_input = 'glove.6B.100d.txt'
word2vec_output = 'glove.6B.100d.word2vec'

glove2word2vec(glove_input,word2vec_output)
#import edilen script çalıştırıldı
#script çalıştırıldığında vektörler değiştirilmiyor, model hala glove ile eğitilmiş durumda

#dosyanın binary olmadığını belirttik
model = KeyedVectors.load_word2vec_format(word2vec_output, binary=False)

#herhangi bir modelin vektörünü bulmak için
istanbul = model['istanbul']
#print(istanbul)

lotr = model.most_similar('gandalf')
#print(lotr)


#king-man + woman = queen
queen = model.most_similar(positive=['woman','king'], negative=['man'], topn=1)
#topn default olarak 10dur, en iyi sonucu istediğimiz için 1 yazdık

print(queen)

#father-man + woman = mother
mother = model.most_similar(positive=['woman','father'], negative=['man'], topn=1)
print(mother)

turkiye = model.most_similar(positive=['ankara', 'germany'], negative=['berlin'], topn=1)
print(turkiye)

teacher = model.most_similar(positive=['teach', 'doctor'], negative=['treat'], topn=1)
#doktordan tedaviyi çıkardığımız zaman elimizde mesleği ifade eden bir vektör kaldı
#bu vektöre öğretmeyi eklediğimizde de öğretmeni elde ettik


