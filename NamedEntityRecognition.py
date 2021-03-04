import nltk

#özel isimleri ve organizasyonları bulmamızı sağlar

text = "Steven Paul Jobs was an American business magnate, industrial designer, investor, and media proprietor. He was the chairman, chief executive officer (CEO), and co-founder of Apple Inc., the chairman and majority shareholder of Pixar"

#texti kelimelerine ayırdık
tokenized = nltk.word_tokenize(text)

#texti ögelerine ayırdık
tagged = nltk.pos_tag(tokenized)

#named entity recognition
named_ent = nltk.ne_chunk(tagged)
#%100 doğruluk vermez fakat iyi çalışır

#alınan named_ent ağaç türündedir. o yüzden print yazmak yerine ağaca bakalım
named_ent.draw()

"""
NAMED ENTITY TÜRLERİ:
    "Türü            Örnek\n",
    "ORGANIZATION    Georgia-Pacific Corp., WHO\n",
    "PERSON          Eddy Bonte, President Obama\n",
    "LOCATION        Murray River, Mount Everest\n",
    "DATE            June, 2008-06-29\n",
    "TIME            two fifty a m, 1:30 p.m.\n",
    "MONEY           175 million Canadian Dollars, GBP 10.40\n",
    "PERCENT         twenty pct, 18.75 %\n",
    "FACILITY        Washington Monument, Stonehenge\n",
    "GPE             South East Asia, Midlothian\n",
"""
