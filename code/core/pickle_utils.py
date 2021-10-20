import pickle


def savePickle(fname: str, data: object):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def loadPickle(fname: str):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data
