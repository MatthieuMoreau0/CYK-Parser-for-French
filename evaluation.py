from PYEVALB import scorer
from PYEVALB import parser

def evaluate(target,predicted):
  target_tree = parser.create_from_bracket_string(target[1:-1]) # first remove first and last bracket
  predicted_tree = parser.create_from_bracket_string(predicted[1:-1]) # first remove first and last bracket
  s = scorer.Scorer()
  result = s.score_trees(target_tree, predicted_tree)

  return result.recall, result.prec
