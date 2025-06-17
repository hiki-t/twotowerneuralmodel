# from  torchtext.data.utils import get_tokenizer
import gensim.downloader as api
from gensim.utils import simple_preprocess

wv = api.load('glove-twitter-25')

wv.get()

wv['']

tokenizer = get_tokenizer("basic_english")

