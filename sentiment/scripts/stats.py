from tass import InterTASSReader
from tass import GeneralTASSReader
from collections import defaultdict

class Stats:

	def __init__(self, reader):
		tweets = list(reader.tweets())  # iterador sobre los tweets
		X = list(reader.X())  # iterador sobre los contenidos de los tweets
		y = list(reader.y())  # iterador sobre las polaridades de los tweets
		tweets_polarity = defaultdict(int)
		for polarity in y:
			tweets_polarity[polarity] += 1
		self._tweets_polarity = dict(tweets_polarity)
		self._tweets_size = len(X)

if __name__ == '__main__':
	print('Basic Statistics')
	print('================')
	print('InterTASS training corpus:')
	stats = Stats(InterTASSReader('TASS/InterTASS/tw_faces4tassTrain1000rc.xml'))
	print('Total number of tweets',stats._tweets_size)
	print('Total numer of tweets per polarity:')
	print('P:', stats._tweets_polarity['P'])
	print('N:', stats._tweets_polarity['N'])
	print('NEU:', stats._tweets_polarity['NEU'])
	print('NONE:', stats._tweets_polarity['NONE'])
	print('================')
	print('GeneralTASS training corpus:')
	stats = Stats(GeneralTASSReader('TASS/GeneralTASS/general-tweets-train-tagged.xml'))
	print('Total number of tweets',stats._tweets_size)
	print('Total numer of tweets per polarity:')
	print('P:', stats._tweets_polarity['P'])
	print('N:', stats._tweets_polarity['N'])
	print('NEU:', stats._tweets_polarity['NEU'])
	print('NONE:', stats._tweets_polarity['NONE'])