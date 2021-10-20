class EarlyStop:
    def __init__(self, patience=5):
        self._patience = patience
        self._min_loss = None
        self._counter = -1

    def __call__(self, loss):
        if self._min_loss is None:
            self._counter = 0
            self._min_loss = loss
            return False

        if self._min_loss > loss:
            self._min_loss = loss
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self._patience
