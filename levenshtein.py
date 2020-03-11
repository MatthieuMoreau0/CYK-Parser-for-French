import numpy as np

def levenshtein(word1,word2):
  """
  Compute Damereau-Levenshtein between two words
  """
  n1,n2=len(word1)+1,len(word2)+1
  dist=np.zeros((n1,n2))

  for i in range(n1):
    dist[i,0]=i
  for j in range(n2):
    dist[0,j]=j

  for i in range(1,n1):
    for j in range(1,n2):
      c = 0 if word1[i-1]==word2[j-1] else 1
      dist[(i,j)]=min(dist[(i-1,j)] +1, #delete
                      dist[(i,j-1)] +1, #insert
                      dist[(i-1,j-1)]+c #substitute
                      )
      if i>1 and j>1 and word1[i-1]==word2[j-2] and word1[i-2]==word2[j-1]:
        transpose_dist=dist[i-2,j-2] + 1
        if transpose_dist<dist[(i,j)]:
          dist[i,j]=transpose_dist #swap / transposition
  
  return dist[-1,-1]

def closest_lexicon_words(word,lexicon):
  """
  Find the closest word in the lexicon using Damereau-Levenshtein distance
  """
  min_dist=len(word)
  closest_words=[]
  for lex_word in lexicon.word2id:
    dist=levenshtein(word,lex_word)
    if dist==min_dist:
      closest_words.append(lex_word)
    if dist<min_dist:
      closest_words=[lex_word]
      min_dist=dist
  return closest_words, min_dist

