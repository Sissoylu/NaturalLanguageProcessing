from nltk.stem import WordNetLemmatizer

#farklı bir kök bulma yöntemi

lem = WordNetLemmatizer()

words = ['eat', 'eating', 'eater', 'eats', 'ate', 'dogs', 'women', 'children']

for word in words:
    print(lem.lemmatize(word))

"""
lemmatizing ile kelimenin sözlükteki anlamına inilir ve çoğullar tekilleştirilebilir.
stemming de children ve women aynı kalmıştı çünkü sözlük anlamına değil sondaki ekine odaklanır.
=lemmatize=
dogs -> dog
women -> woman
children -> child
"""

#eğer kelimenin fiil olduğunu belirtirsek, lemmatizing işleminde kelimenin kökünü bu şekilde bulabilir.
# 'v' => verb
print(lem.lemmatize('eating','v'))
print(lem.lemmatize('ate', 'v'))