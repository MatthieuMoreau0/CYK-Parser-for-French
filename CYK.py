import numpy as np
from nltk import Tree, Nonterminal

def build_tree_from_CYK(S,words):
  """
  This function computes the tree using the array build by the CYK algorithm
  """
  root_tag=None
  n=len(words)

  for tag in S[n-1,0]:
    if "SENT" in tag.symbol():
      root_tag=tag      
      
  if root_tag is None:
    print(f"Can't parse sentence : {' '.join(words)}")
    return None

  def aux_rec(tag,ind):
    left_child=S[ind][tag]['left_child']
    left_tag=S[ind][tag]["left_tag"]
    right_child=S[ind][tag]["right_child"]
    right_tag=S[ind][tag]["right_tag"]
    if(left_child is None or right_child is None):
      return Tree(tag.symbol(),[words[ind[1]]])
    return Tree(tag.symbol(),[aux_rec(left_tag,left_child),aux_rec(right_tag,right_child)])

  return aux_rec(root_tag,(n-1,0))
  


def CYK(sentence,prod_dict,lexicon,OOV):
  """
  THis function parses a sentence using a pcfg, a lexicon and an OOV module
  """
  print(f"Parsing the sentence : {sentence}.")
  words=sentence.split(" ")
  n=len(words)
  S=np.empty((n,n),dtype=object)


  # We compute for each word the known tags is is associated to in our lexicon with their respective probabilities
  for (i,word) in enumerate(words):
    lexicon_token=lexicon.get_token(word)
    if lexicon_token is not None:
      tags=lexicon_token.get_tags()
    else:
      tags=OOV.get_OOV_tag(word)
    S[0,i]={}
    for tag in tags:
      S[0,i][tag]={"log_prob":np.log(tags[tag]), "left_child":None, "left_tag":None, "right_tag":None, "right_child":None}
  
  for i in range(1,n): #length of the substring -1
    for j in range(n-i): #position of start of the substring
      S[i,j]={}
      for k in range(j+1,i+j+1): #position of split
        first_part_tags=S[k-j-1,j]
        second_part_tags=S[i-(k-j),k]
        for tag1 in first_part_tags:
          for tag2 in second_part_tags:
            if((tag1,tag2) not in prod_dict):
              prods=[]
            else:
              prods=prod_dict[(tag1,tag2)]
            for prod in prods:
              log_prob=np.log(prod.prob())+first_part_tags[tag1]["log_prob"]+second_part_tags[tag2]["log_prob"]
              if prod.lhs() in S[i,j]:
                if S[i,j][prod.lhs()]["log_prob"]<log_prob:
                #if several paths use the same rule, we retain the path with max probability
                  S[i,j][prod.lhs()]={"log_prob": log_prob, 
                                      "left_child": (k-j-1,j),
                                      "right_child": (i-(k-j),k),
                                      "left_tag":tag1,
                                      "right_tag":tag2}
              else:
                #we store the log probabilities to avoid numerical errors when computing the product of probabilities
                S[i,j][prod.lhs()]={"log_prob": log_prob,
                                      "left_child": (k-j-1,j),
                                      "right_child": (i-(k-j),k),
                                      "left_tag":tag1,
                                      "right_tag":tag2}
  

  tree=build_tree_from_CYK(S,words)
  return tree




