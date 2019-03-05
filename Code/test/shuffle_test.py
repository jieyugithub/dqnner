from random import shuffle

int2tags = ['111', '222', '333', '444']
print int2tags
shuffle(int2tags)
print int2tags
shuffle(int2tags)
print int2tags

newArticles  = ['1', '22', '333', '4444']
shuffledIndxs = [range(len(q)) for q in newArticles]
print shuffledIndxs