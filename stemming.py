from nltk.stem import PorterStemmer

porters = PorterStemmer()
#kelimenin kökünü almaya => stemming denir

words = ['eat', 'eating', 'eater', 'eats', 'ate', 'dogs', 'women', 'children']

for i in words:
    print(porters.stem(i))

#stemming yapılırken sadece sondaki ekleri dikkate aldığı için
#ate, women, children, eater değiştirilmeden kaldı.
#eater tamamen farklı bir anlama sahip olduğu için değişmemesi doğru olan


