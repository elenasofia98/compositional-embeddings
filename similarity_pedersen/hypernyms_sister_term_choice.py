from nltk.corpus import wordnet as wn, wordnet_ic
from nltk.corpus.reader import information_content

""" On Inheritance problem
The sister term choice is lead by the choice of a given father for the synset we are looking at.
In wordnet multiple inheritance is allowed so we should define which father is taken as 'test' father 
or take them all and later randomly select the 'test' one.

The first idea is that the deepest node (among other fathers) is a feasible 'test' father 
cause it is probably the most specific and most informative one.

What if we simply take the first of hypernyms() given a certain synset? (For simplicity)
Here the results:

POS only_synset_with_multiple_hyper   #_right_classification  #_wrong_classification
n	False	82115	73865	524
n	True	82115	900	522
v	False	13767	13195	13
v	True	13767	18	13



The second idea is the following. We would like is the most informative hyper.
This definition can be used calculating the "informative power" of a word according to some information content model
However his "informative power" can be influenced by the chosen ic model
ic-brown.dat
POS only_synset_with_multiple_hyper   #_right_classification  #_wrong_classification
n	False	82115	73721	668
n	True	82115	755	667
v	False	13767	13187	21
v	True	13767	10	21
"""


def first_equals_most_informative(pos, ic, only_synset_with_multiple_hyper=False):
    wrong = 0
    right = 0
    tot = 0

    if only_synset_with_multiple_hyper:
        min_len = 1
    else:
        min_len = 0

    for syn in wn.all_synsets(pos=pos):
        tot += 1
        hyper_paths = syn.hypernym_paths()
        hyper_syns = syn.hypernyms()

        if len(hyper_paths) > min_len and len(hyper_syns) > min_len:
            most_informative_hyper_path = max(hyper_paths, key=lambda x: information_content(x[-2], ic))

            if hyper_syns[0].name() != most_informative_hyper_path[-2].name():
                wrong += 1
            else:
                right += 1

    return '\t'.join([pos, str(only_synset_with_multiple_hyper), str(tot), str(right), str(wrong)])




def first_equals_max_depth_hyper(pos, only_synset_with_multiple_hyper):
    wrong = 0
    right = 0
    tot = 0

    if only_synset_with_multiple_hyper:
        min_len = 1
    else:
        min_len = 0

    for syn in wn.all_synsets(pos=pos):
        tot += 1
        hyper_paths = syn.hypernym_paths()
        hyper_syns = syn.hypernyms()

        if len(hyper_paths) > min_len and len(hyper_syns) > min_len:
            max_depth_hyper = max(hyper_paths, key=len)

            if hyper_syns[0].name() != max_depth_hyper[-2].name():
                wrong += 1
            else:
                right += 1

    return '\t'.join([pos, str(only_synset_with_multiple_hyper), str(tot), str(right), str(wrong)])


print(first_equals_max_depth_hyper('n', only_synset_with_multiple_hyper=False))
print(first_equals_max_depth_hyper('n', only_synset_with_multiple_hyper=True))
print(first_equals_max_depth_hyper('v', only_synset_with_multiple_hyper=False))
print(first_equals_max_depth_hyper('v', only_synset_with_multiple_hyper=True))


INFORMATION_CONTENT = wordnet_ic.ic('ic-brown.dat')
print(first_equals_most_informative('n', ic=INFORMATION_CONTENT, only_synset_with_multiple_hyper=False))
print(first_equals_most_informative('n', ic=INFORMATION_CONTENT, only_synset_with_multiple_hyper=True))
print(first_equals_most_informative('v', ic=INFORMATION_CONTENT, only_synset_with_multiple_hyper=False))
print(first_equals_most_informative('v', ic=INFORMATION_CONTENT, only_synset_with_multiple_hyper=True))