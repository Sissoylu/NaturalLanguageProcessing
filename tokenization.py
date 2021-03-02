from nltk.tokenize import sent_tokenize, word_tokenize

#metni kelimelerine ayırmak => tokenleştirme

text = "Alan Turing, İngiliz matematikçi, bilgisayar bilimcisi ve kriptolog. Bilgisayar biliminin kurucusu sayılır. Geliştirmiş olduğu Turing testi ile makinelerin ve bilgisayarların düşünme yetisine sahip olup olamayacakları konusunda bir kriter öne sürmüştür."

"""
print(text.split())
print("\n")
print(word_tokenize(text)) #noktalama işaretlerini de bir kelime olarak kabul ediyor.
print("\n")
print(sent_tokenize(text)) #cümleler tokenleştirildi
"""

for token in word_tokenize(text):
    print(token)