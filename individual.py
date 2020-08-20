





class ListWithAttributes(list):
    def __new__(self, *args, **kwargs):
            return super(ListWithAttributes, self).__new__(self, args, kwargs)
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
