# Parser for French

This project is my work for the second assignment the course Algorithm for Speech and Natural Language Processing of the MVA. The goal is to implement a probabilistic parser for French.

## Dependencies

This code has been tested on Python 3.7.4.

We use f-strings, so Python>=3.6 is required.

Necessary packages : nltk, numpy, pyevalb, pandas, pickle.

## Usage

Run the `run.sh` script to call the main.py module. It can be used in 2 ways:

- `./run.sh --dataset wanted_dataset` to evaluate the parser on the wanted dataset sentences. Two possible choices : `dev` or `test`
- `./run.sh --i "Your sentences to parse"` to run the parser on custom sentences. Each token has to be separated by a single whitespace. Your input can take several lines with one setence per line.

You can run get help and more detail on each command by typing :

```bash
./run.sh --h
```

## Output

The resulting parsed sentences are written in an output file `*.output_parser`, whose full name is printed when the script is ran.

When a sentence cannot be parsed it is written as is in the output file.

## Examples

```bash
./run.sh --i "C'est la vie ."
```

```bash
./run.sh --dataset test
```

```bash
./run.sh --dataset dev
```
