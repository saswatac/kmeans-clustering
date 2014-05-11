__author__ = 'saswatac'


from sklearn.feature_extraction.text import TfidfVectorizer



class StemmedTfidfVectorizer(TfidfVectorizer):
    def __init__(self, min_df, max_df, stop_words, charset_error, stemmer):
        super(StemmedTfidfVectorizer, self).__init__(min_df=min_df, max_df=max_df, stop_words=stop_words, charset_error=charset_error)
        self.stemmer = stemmer

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))







