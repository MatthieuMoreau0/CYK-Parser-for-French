import numpy as np
from operator import itemgetter
from levenshtein import closest_lexicon_words
from lexicon import *
import pickle

def case_normalizer(word, dictionary):
  """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
  w = word
  lower = (dictionary.get(w.lower(), 1e12), w.lower())
  upper = (dictionary.get(w.upper(), 1e12), w.upper())
  title = (dictionary.get(w.title(), 1e12), w.title())
  results = [lower, upper, title]
  results.sort()
  index, w = results[0]
  if index != 1e12:
    return w
  return word

def normalize(word, word_id):
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word

class OOV():
  """
  Module to handle out of vocabulary words
  """
  def __init__(self,lexicon):
    words, embeddings = pickle.load(open('./polyglot-fr.pkl', 'rb'), encoding='latin1')
    self.embeddings=(embeddings.T/np.linalg.norm(embeddings,axis=1)).T
    self.word_id = {w:i for (i, w) in enumerate(words)}
    self.id_word = dict(enumerate(words))
    self.lexicon=lexicon
    self.OOV_tags={}

  def cosine_nearest(self,word, k=10, verbose=False):
    org_word=word
    word=normalize(word,self.word_id)
    if not word:
      if(verbose):
        print(f"No embeddings found for {org_word}")
      return []
    word_index=self.word_id[word]
    distances = (((self.embeddings - self.embeddings[word_index]) ** 2).sum(axis=1))
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    return [self.id_word[sorted_distances[i][0]] for i in range(k)]

  def get_OOV_tag(self,word, verbose=False):
    #if we have already studied that OOV word, we return the stored associated tags
    if(word in self.OOV_tags):
      return self.OOV_tags[word]
    
    nearest_words=self.cosine_nearest(word,k=10)
    #first we look for similar embeddings and return the POS of the most mist similar embedding that can be found in the lexicon
    lex_words=[]
    words=[]
    for _word in nearest_words:
      lex_word=self.lexicon.get_token(_word)
      #we look for the word in the lexicon
      if(lex_word is not None):
        lex_words.append(lex_word.get_tags())
        words.append(lex_word)
        #remove this break if we want to associate several words
        # break
    if len(lex_words)>0:
      if(verbose):
        print(f"OOV word '{word}' associated to '{words}' using embeddings")
      #we store the associated tags to make future computations faster if that word is found again
      self.OOV_tags[word]=sum_tags(lex_words)
      return self.OOV_tags[word]

    
    #if we end up here, there could be 2 reasons :
    #    either the word is not in the embeddings, so we couldn't find similar words
    #    or none of the nearest words were found in the vocabulary
    #in this case we fallback to computing the closest words in the lexicon using damereau-levenshtein distance
    closest_words, min_dist = closest_lexicon_words(word,self.lexicon)
    if verbose:
      print(f"OOV word '{word}' closest words are {closest_words}")
    for word in closest_words:
      lex_words.append(self.lexicon.get_token(closest_words[0]).get_tags())
    self.OOV_tags[word]=sum_tags(lex_words)
    return self.OOV_tags[word]
