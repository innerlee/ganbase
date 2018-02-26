class LoopLoader(object):
    def __init__(self, loader):
        #region yapf: disable
        self.loader = loader
        self.iter   = iter(loader)
        self.n      = len(loader)
        self.i      = 0
        #endregion yapf: enable

    def __iter__(self):
        return self

    def __len__(self):
        return self.n

    def __next__(self):
        if self.i == self.n:
            self.iter = iter(self.loader)
            self.i = 0
        self.i += 1
        return next(self.iter)
