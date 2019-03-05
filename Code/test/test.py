def splitBars(w):
    return [q.strip() for q in w.split('|')]


sb = splitBars("a|b|c| d |e")

print sb

q = " a "
print '['+q+']', '['+q.strip()+']'

q = ' a b '
print '['+q+']', '['+q.strip()+']'