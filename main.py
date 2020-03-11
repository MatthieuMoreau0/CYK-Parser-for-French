from nltk import Tree, Nonterminal, Production
from nltk.treetransforms import chomsky_normal_form, collapse_unary
from lexicon import LexiconToken, Lexicon
from CYK import CYK
from OOV import OOV
from evaluation import evaluate
import nltk
import numpy as np
import time
import random
import functools
from multiprocessing import Pool, cpu_count
import argparse

def remove_functionnal_labels(tree):
  """
  Strip trees from functionnal labels
  """
  if isinstance(tree,Tree):
    if "-" in tree.label():
      tree.set_label(tree.label()[:tree.label().find("-")])

    for child in tree:
      remove_functionnal_labels(child)

def get_lexicon(productions):
  """
  Returns the probabilistic lexicon extracted from a list of productions
  """
  lexicon_dic={}
  for production in productions:
    if production.is_lexical():
      rhs=production.rhs()[0]
      if(rhs not in lexicon_dic):
        lexicon_dic[rhs]=[]
      symbol=production.lhs().symbol()
      lexicon_dic[rhs].append(Nonterminal(symbol))
  lexicon=Lexicon(lexicon_dic)
  return lexicon

def multiprocess_func(func,n_jobs,arg):
  """
  Runs a function on multiple threads
  """
  if(n_jobs == -1):
    n_jobs=cpu_count()
  start=time.time()
  with Pool(n_jobs) as p:
    result = p.map(func,arg)
  print(f"Finished parsing in {time.time() - start}s")
  return result

def split_dataset(path):
  """
  Split the dataset in a training, dev and test set
  """
  productions=[]
  dev_sentences=[]
  dev_parsing=[]
  test_sentences=[]
  test_parsing=[]
  with open(path,'r',encoding="utf-8") as f:
    lines=f.readlines()
    n=len(lines)
    train_lines=lines[:int(0.9*n)]
    random.shuffle(train_lines)
    test_lines=lines[int(0.9*n):]
    for sentence in train_lines[:int(0.8*n)]:
      t=Tree.fromstring(sentence)
      t=t[0] #ignore the extra parenthesis from the dataset
      remove_functionnal_labels(t)
      chomsky_normal_form(t,horzMarkov=2)
      #we use & as unary char because there is a POS tag that uses + already
      collapse_unary(t,collapsePOS=True,joinChar="&")
      productions.extend(t.productions())
    for sentence in train_lines[int(0.8*n):]:
      t=Tree.fromstring(sentence)
      remove_functionnal_labels(t)
      dev_parsing.append(t)
      dev_sentences.append(' '.join(t.flatten()))
    for sentence in test_lines:
      t=Tree.fromstring(sentence)
      remove_functionnal_labels(t)
      test_parsing.append(t)
      test_sentences.append(' '.join(t.flatten()))
  
  return productions, dev_sentences, dev_parsing, test_sentences, test_parsing

#Arguments when running the script
parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=False, help="Input to parse")
parser.add_argument("--dataset", choices=["dev", "test"],
                    help="Dataset to evaluate (default: test)", default="test")
args = parser.parse_args()


if __name__=="__main__":
  productions, dev_sentences, dev_parsing, test_sentences, test_parsing = split_dataset("sequoia-corpus+fct.mrg_strict")

  lexicon=get_lexicon(productions)
  productions=[production for production in productions if not production.is_lexical()]

  pcfg=nltk.induce_pcfg(Nonterminal("SENT"),productions)

  # we compute a production dictionnary so that we can find very fast all productions given a right hand side, this will make the CYK algorithm run faster
  prod_dict={}
  for production in pcfg.productions():
    if production.rhs() not in prod_dict:
      prod_dict[production.rhs()]=[]
    prod_dict[production.rhs()].append(production)

  OOV_module=OOV(lexicon)
  parsed_tree,parsed_tree_output=[],[]
  total_recall,total_prec=[],[]

  #auxiliary function to multithread the computation of CYK
  CYK_m = functools.partial(CYK, prod_dict=prod_dict, lexicon=lexicon, OOV=OOV_module)

  #select the sentences to parse according to the arguments
  if args.input:
    sentences=args.input.split('\n')
    sentences_to_parse=sentences
    expected_parsing=None
  
  else:
    if args.dataset=="test":
      expected_parsing=test_parsing
      sentences_to_parse=test_sentences
    if args.dataset=="dev":
      expected_parsing=dev_parsing
      sentences_to_parse=dev_sentences
  parsed_sentences=0
  result_CYK = multiprocess_func(CYK_m,-1,sentences_to_parse)


  #parse all sentences
  for (i,res) in enumerate(result_CYK):
    if res is None:
      parsed_tree_output.append(sentences_to_parse[i])
    else:
      res.set_label('SENT')
      res.un_chomsky_normal_form(unaryChar="&")
      res=Tree('',[res])
      parsed_tree.append(res)
      parsed_tree_output.append(res.pformat(margin=100000))
      if expected_parsing is not None:
        recall,prec=evaluate(expected_parsing[i].pformat(margin=1000000),parsed_tree_output[-1])
        total_recall.append(recall)
        total_prec.append(prec)
      parsed_sentences+=1
  
  print(f"Managed to parse {parsed_sentences}/{len(result_CYK)} sentences.")

  if expected_parsing is not None:
    print(f"Total Recall : {np.mean(total_recall)} - Total Precision : {np.mean(total_prec)}")

  if args.input:
    output_string="custom_input"
  elif args.dataset=="test":
    output_string="evaluation_data"
  elif args.dataset=="dev":
    output_string="dev_data"

  output_file=output_string+".parser_output"

  print(f"Writing output result in file : {output_file}")
  #write output
  with open(output_file,'w',encoding="utf-8") as f:
    for tree in parsed_tree_output:
      f.writelines(tree)
      f.writelines("\n")

  print("Done")


