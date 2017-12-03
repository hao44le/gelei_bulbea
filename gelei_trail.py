import bulbea as bb
share = bb.Share('WIKI', 'GOOGL')
from bulbea.learn.evaluation import split
Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)
