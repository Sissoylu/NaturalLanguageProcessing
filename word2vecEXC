from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
import re

paragraf = "Fakat, Allah kahretsin, insan anlatmak istiyor albayım; böyle budalaca bir özleme kapılıyor. Bir yandan da hiç konuşmak istemiyor. Tıpkı oyunlardaki gibi çelişik duyguların altında eziliyor. Fakat benim de sevmeğe hakkım yok mu albayım? Yok. Peki albayım. Ben de susarım o zaman. Gecekondumda oturur, anlaşılmayı beklerim. Fakat albayım, adresimi bilmeden beni nasıl bulup anlayacaklar? Sorarım size: Nasıl? Kim bilecek benim insanlardan kaçtığımı? Ben ölmek istiyorum sayın albayım, ölmek. Bir yandan da göz ucuyla ölümümün nasıl karşılanacağını seyretmek istiyorum. Tehlikeli oyunlar oynamak istiyor insan; bir yandan da kılına zarar gelsin istemiyor. Küçük oyunlar istemiyorum albayım."

paragraf = paragraf.lower()

cumle_list = tokenize.sent_tokenize(paragraf)
#print(cumle_list[:5])


cumle_noktalamasiz = [re.sub(r'[^\w\s]', '', x) for x in cumle_list]
#print(cumle_noktalamasiz[:5])

paragraf_duzenlenmis = " ".join(cumle_noktalamasiz)
#print(paragraf_duzenlenmis[:50])

word2vec_list = [[x] for x in cumle_noktalamasiz]
print(word2vec_list[:5])

#noktalama işaretlerinden kurtulduk
#tokenizer = RegexpTokenizer(r'\w+')
#a = tokenizer.tokenize(paragraf)

#paragraf_duzenlenmis = " ".join(a)
#print(paragraf_duzenlenmis[:50])

#print("paragraf_duzenlenmisteki unique kelime sayisi", len(set(paragraf_duzenlenmis.split())))
#print("paragraf_duzenlenmisteki total kelime sayısı", len(paragraf_duzenlenmis.split()))

#word2vec_list = [[x] for x in a]
#print(word2vec_list[:5])
