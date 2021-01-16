from nltk.corpus import wordnet as wn

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
"""


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