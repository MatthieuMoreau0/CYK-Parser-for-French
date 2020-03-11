import pandas as pd
from nltk import Nonterminal

class Lexicon:
  """
  Class for the lexicon
  """
  def __init__(self, lexicon_dic):
    self.lex=[]
    self.word2id={}
    for lexicon_token in lexicon_dic:
      self.lex.append(LexiconToken(lexicon_token,lexicon_dic[lexicon_token]))
      self.word2id[lexicon_token]=len(self.lex)-1
  
  def get_token(self,token):
    #handle different capitalization of the word
    if token in self.word2id:
      #first we try the word with the text capitalization
      return self.lex[self.word2id[token]]
    if( token.lower()  in self.word2id):
      return self.lex[self.word2id[token.lower()]]
    if( token.capitalize()  in self.word2id):
      return self.lex[self.word2id[token.capitalize()]]
    return None


def sum_tags(list_tags):
  """
  Returns the probability of each POS for a list of words
  """
  sum_tag={}
  for tag in list_tags:
    for pos in tag:
      if not pos in sum_tag:
        sum_tag[pos]=0
      sum_tag[pos]+=tag[pos]
  for pos in sum_tag:
    sum_tag[pos]/=len(list_tags)
  return sum_tag





class LexiconToken:
  """
  Class for a token. Associate each possible tag to its probability
  """
  def __init__(self,symbol,tags=None):
    self._symbol=symbol
    self._tags={}
    total=0
    if(type(tags) is list):
      for tag in tags:
        if tag not in self._tags:
          self._tags[tag]=0.
        self._tags[tag]+=1
        total+=1
      #normalization
      for tag in self._tags:
        self._tags[tag]/=total
      keys=list(self._tags.keys())
      for tag in keys:
        if '&' in tag.symbol():
          ind=tag.symbol().rfind('&')
          symbol1=tag.symbol()[:ind]
          symbol2=tag.symbol()[ind+1:]
          if(Nonterminal(symbol1) not in self._tags):
            self._tags[Nonterminal(symbol1)]=0
          if(Nonterminal(symbol2) not in self._tags):
            self._tags[Nonterminal(symbol2)]=0
          self._tags[Nonterminal(symbol1)]+=self._tags[tag]/6.
          self._tags[Nonterminal(symbol2)]+=self._tags[tag]/6.
          self._tags[tag]=4*self._tags[tag]/6.


  def __str__(self):
    return f"{self._symbol} : {self._tags}"
  
  def __repr__(self):
    return f"{self._symbol} : {self._tags}"

  def get_tags(self):
    return self._tags
  
  def get_symbol(self):
    return self._symbol
      


