from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "The series revolves around two brothers, Lincoln Burrows (Dominic Purcell) and Michael Scofield (Wentworth Miller); Burrows has been sentenced to death for a crime he did not commit, and Scofield devises an elaborate plan to help his brother escape prison and clear his name."

#list of all stopwords using in english
#print(stopwords.words('english'))

#print(stopwords.words('turkish'))

#ingilizcede kullanılan stopwordleri aldık
stopwords = stopwords.words('english')

#textimizi kelimelerine ayırdık
words = word_tokenize(text)

filtered_words = []

#kelime stopwordlerde bulunmuyorsa yeni listeye bu kelimeyi ekle
#bu sayede cümleyi stopwordlerden arındırmış olduk
for word in words:
    if word not in stopwords:
        filtered_words.append(word)

print(filtered_words)