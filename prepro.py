

def filter(words):
	new_words = []
	for word in words:
		if '@' not in word and 'http' not in word and '#' not in word:
			new_words.append(word.lower())

	return new_words
lines = open('normalized_tweets.txt','r').readlines()
fw = open('norm_tweets.txt', 'w')
for line in lines:
	try:
		key, value = line.strip().split('\t')
		key_words = key.strip().split(' ')
		value_words = value.strip().split(' ')
		key_words = filter(key_words)
		value_words =filter(value_words)
		for word in key_words:
			if "_" not in word:
				fw.write(word)
			else:
				words_a =  word.split('_')
				words_n = ' '.join(words_a)
				fw.write(words_n)
			if key_words.index(word)<len(key_words)-1:
				fw.write(" ")
		fw.write("\t")
		for word in value_words:
			if "_" not in word:
				fw.write(word)
			else:
				words_a =  word.split('_')
				words_n = ' '.join(words_a)
				fw.write(words_n)
			if value_words.index(word)<len(value_words)-1:
				fw.write(" ")
		fw.write("\n")
	except Exception,e:
		pass