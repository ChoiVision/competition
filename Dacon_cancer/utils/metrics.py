from sklearn import metrics

def score(true, pred, average='marco'):
    score= metrics.f1_score(true, pred, )