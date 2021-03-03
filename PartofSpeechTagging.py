import nltk

#kelimeyi ögelerine ayirma

text = "Stephen Edwin King is an American author of horror, supernatural fiction, suspense, crime, science-fiction, and fantasy novels"

#önce kelimelere ayırmalıyız
tokenized = nltk.word_tokenize(text)

#.pos_tag() tüm kelimelerin ayrı ayrı ne olduğunu yazar
print(nltk.pos_tag(tokenized))

"""
"CC     coordinating conjunction\n",
"CD     cardinal digit\n",
"DT     determiner\n",
"EX     existential there (like: \"there is\" ... think of it like \"there exists\")\n",
"FW     foreign word\n",
"IN     preposition/subordinating conjunction\n",
"JJ     adjective 'big'\n",
"JJR    adjective, comparative 'bigger'\n",
"JJS    adjective, superlative 'biggest'\n",
"LS     list marker 1)\n",
"MD     modal could, will\n",
"NN     noun, singular 'desk'\n",
"NNS    noun plural 'desks'\n",
"NNP    proper noun, singular 'Harrison'\n",
"NNPS   proper noun, plural 'Americans'\n",
"PDT    predeterminer 'all the kids'\n",
"POS    possessive ending parent's\n",
"PRP    personal pronoun I, he, she\n",
"PRP$   possessive pronoun my, his, hers\n",
"RB     adverb very, silently,\n",
"RBR    adverb, comparative better\n",
"RBS    adverb, superlative best\n",
"RP     particle give up\n",
"TO     to go 'to' the store.\n",
"UH     interjection errrrrrrrm\n",
"VB     verb, base form take\n",
"VBD    verb, past tense took\n",
"VBG    verb, gerund/present participle taking\n",
"VBN    verb, past participle taken\n",
"VBP    verb, sing. present, non-3d take\",
"VBZ    verb, 3rd person sing. present takes",
"WDT    wh-determiner which",
"WP     wh-pronoun who, what",
"WP$    possessive wh-pronoun whose",
"""