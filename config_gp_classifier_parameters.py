c = 3
assert c >= 3, 'c decides probability values for train positives and negative values.' \
               ' Any value less than 3 gives not so appropriate probabilities.'
#
bias = -0.9*c

