from __future__ import unicode_literals, print_function
#import pandas as pd
import random
from pathlib import Path
import spacy
import openpyxl
import re
from spacy.training.example import Example

from SpacyPreprocessing import *
import json


#========================================================================

model_input_dir = None
model_output_dir = Path("C:\\Users\\arikg\\PycharmProjects\\EntityExtrcat\\models\\Model_conll_31_01")

input_files = ['./Data/conll2003_spacy/train.json',  './Data/conll2003_spacy/valid.json','./Data/conll2003_spacy/test.json','./Data/all_tweets/train/train.xlsx']

#C:/Users/arikg/PycharmProjects/EntityExtrcat/models/Model

INPUT_TYPE = 'TWEETS'
#INPUT_TYPE = 'FORUMS'
#=====================================================================

# example of train data structure
'''
TRAIN_DATA = [
    ('Today Tesla released a new car', { 'entities': [(6, 11, 'ORG')]
    }),
   # ('012345678901234567890123456789012345678901234'
     ('Also SixGill is a cyber security company', {
        'entities': [(5, 12, 'ORG')]
    })
]
'''

# --------------------------------------------------------------
#  prepare train data from excel
# --------------------------------------------------------------


def get_train_data_from_excel(filename):

    col_list = ["es_id", "content", 'baseline']
    df = pd.read_excel(filename, usecols=col_list)
    org_l = df['baseline'].astype(str).tolist()  #  Non clean orgs as defined at the original file
    content_l = df['content'].astype(str).tolist()
    id_l = df['es_id'].astype(str).tolist()

    TRAIN_DATA_T = []
    last_id = ''

    # STOPSET = {'-',' ','$', '(',')','[',']','\n','"','_',';','#','.'}
    STOPSET = {' ', '\n', "'"}  # {' ','(','[','\n',';','.','"',':'}
    STARTSET = {' ', '\n', "'"}  # '$',')',']','\n',';','#','@',' ','"',':'}
    txt = ''
    print("loading and preparing file: ", filename)
    for i, id in enumerate(id_l):
        ent_lst = []
        txt = str(content_l[i])
        txt =  text_preprocessing(txt,'NONE')
        orgs = str(org_l[i])
        orgs = orgs.replace("[", "")
        orgs = orgs.replace("]", "")
        orgs = orgs.replace('\'', "")
        orgs = orgs.split(",")
        orgs = [x for x in orgs if x != '' and x != ' ']

        if orgs:
            new_orgs = []
            for x in orgs:
                tmp = org_name_cleanup(x, False)
                if tmp != '' and tmp not in black_list_dict:
                    new_orgs.append(x)
            orgs = new_orgs
        if orgs:
            for org in orgs:
                loc = 0
                endloc = 0
                try:
                    locs = re.finditer(org, txt)
                except:
                    print(org)
                    break
                locs = [match.start() for match in locs]

                for loc in locs:
                    endloc = int(loc) + len(org)
                    # ent_lst.append((loc, endloc, 'ORG'))
                    if (len(txt) == endloc or txt[endloc] in STOPSET) and (loc == 0 or txt[loc - 1] in STARTSET):
                        ent_lst.append((loc, endloc, 'ORG'))
        if ent_lst:
            TRAIN_DATA_T.append((txt, {'entities': ent_lst}, id))
    print(filename, "  ----  ", len(TRAIN_DATA_T))
    return TRAIN_DATA_T





def get_train_data_from_json(filename):
    f = open(filename, 'r')
    tdata  = json.load(f)
    tdata = tdata[0]['paragraphs']
    TRAIN_DATA_J = []
    orgid = 0
    fname = filename
    loc = 0
    while loc > -1:
        loc = fname.find('/')
        fname = fname[loc+1:]
    fname = fname[:-5]
    ent_lst = []
    # STOPSET = {'-',' ','$', '(',')','[',']','\n','"','_',';','#','.'}
    STOPSET = {' ', '\n', "'"}  # {' ','(','[','\n',';','.','"',':'}
    STARTSET = {' ', '\n', "'"}  # '$',')',']','\n',';','#','@',' ','"',':'}

    for prg in tdata:
        snts = prg['sentences']

        for snt in snts:
            txt = []
            ents = []
            tkns = snt['tokens']
            phrs = []
            for t in tkns:
                tg = t['tag']
                ort = t['orth']
                nr = t['ner']
                if ort == "-DOCSTART-":
                    txt = []
                    break

                if nr in {'I-ORG', 'L-ORG', 'B-ORG', 'U-ORG'}:
                    if tg in ['.', ',', 'POS', "\""]:
                        if phrs:
                            phrs[-1] = phrs[-1][:-1]
                        elif txt and len(txt[-1] > 1):
                            txt[-1] = txt[-1][:-1]
                    phrs.append(ort)
                    if tg not in {'\"', "''"}:
                        phrs[-1]= phrs[-1]+' '

                    if nr == 'U-ORG':
                        ents.append(''.join(phrs).strip())
                        txt.append(''.join(phrs))
                        phrs = []
                    elif nr == 'L-ORG':
                        ents.append(''.join(phrs).strip())
                        txt.append(''.join(phrs))
                        phrs = []
                else:
                    if tg in ['.', ',', 'POS', "\""]:
                        if txt and len(txt[-1]) > 1:
                            txt[-1] = txt[-1][:-1]
                    txt.append(ort)
                    if tg not in {'\"', "''"}:
                        txt[-1] = txt[-1]+' '

                    if nr == 'U-ORG':
                        ents.append(phrs[0].strip())
                        txt.append(phrs[0])
                        phrs = []
                    elif nr == 'L-ORG':
                        ents.append(''.join(phrs).strip())
                        txt.append(''.join(phrs))
                        phrs = []

            if txt and ents:
                txt = ''.join(txt)
                ent_lst = []
                for org in ents:
                    loc = 0
                    endloc = 0
                    try:
                        locs = re.finditer(org, txt)
                    except:
                        print(org)
                        break
                    locs = [match.start() for match in locs]
                    for loc in locs:
                        endloc = loc + len(org)
                        # ent_lst.append((loc, endloc, 'ORG'))
                        if (len(txt) == endloc or txt[endloc] in STOPSET) and (
                                loc == 0 or txt[loc - 1] in STARTSET):
                            ent_lst.append((loc, endloc, 'ORG'))
                            stop = True

                if ent_lst:
                    TRAIN_DATA_J.append((txt, {'entities': ent_lst},"J-"+fname+'-'+str(orgid)))
                    orgid+= 1

    print(filename, "  ----  ", len(TRAIN_DATA_J))
    return TRAIN_DATA_J


def model_train(td, n_iter, model_in, model_out):

    tdln = len(td)
    print('TRAIN_DATA len: ', tdln)

    # load the model
    ALL_PIPES = ['tagger', 'parser', 'ner', 'lemmatizer', 'textcat']

    if model_in is not None:
        nlp = spacy.load(model_in)
        print("Loaded model '%s'" % model_in)
    else:
        #nlp = spacy.blank("en")
        nlp = spacy.load("en_core_web_sm")

    # set up the pipeline

    if 'ner' not in nlp.pipe_names:
        # ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner', last=True)

    ner = nlp.get_pipe('ner')
    optimizer = nlp.create_optimizer()

    indx = 0
    for _, annotations, _ in td:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # Here, we want to train the recognizer by disabling the unnecessary pipeline except for NER.
    #other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    #with nlp.disable_pipes(*other_pipes):  # only train NER

    bad_ids = []
    bad_txt = []
    bad_anot = []
    itn = 0
    bad_count = 0
    while itn < n_iter:
        random.shuffle(td)
        losses = {}
        good_count = 0
        for batch in spacy.util.minibatch(td, size=2):
            for text, annotations, id in batch:
                if id not in bad_ids:
                    try:
                        doc = nlp.make_doc(text)
                        try:
                            example = Example.from_dict(doc, annotations)
                            # Update the model
                            nlp.update([example], losses=losses, sgd=optimizer, drop=0.5)
                            good_count += 1
                        except:
                            if itn == 0:
                                bad_ids.append(id)
                                bad_txt.append(text)
                                bad_anot.append(annotations)
                                bad_count += 1
                                if bad_count < (tdln/10):
                                    print('problem2 at ---> ', id)
                                else:
                                    print("Aborted")
                                    exit(0)
                    except:
                        if itn == 0:
                            bad_count += 1
                            if bad_count < (tdln/10):

                                print('problem1 at ---> ', id)
                            else:
                               print("Aborted")
                               exit(0)
                else:
                    if itn == 0:
                        print("skipped id", id)
                if indx % 100 == 0 and itn == 0:
                    print(indx)
                indx +=1

        if itn % 2 == 0:
            try:
                with open("Num_Iterations.txt", "r") as a_file:
                    limit = a_file.read()
                    if int(limit) < n_iter:
                        n_iter = int(limit)
                        a_file.close()
            except:
                pass

        if itn == 0:
            print('Good Trains: ', good_count)
            if itn == 0:
                bad_ids = pd.DataFrame(bad_ids, columns=["ids"])
                bad_ids['text'] = bad_txt
                bad_ids['anotations'] = bad_anot
                bad_ids.to_excel('./Data/bad_ids.xlsx')

        print("interation: ", itn, losses)
        itn += 1

        #    if itn%5 == 0:
        #       val = input("Stop: [y,n]")
        #  if val =='y' or val == 'Y':
        #     break


    if model_out is not None:
        model_output_dir = Path(model_out)
        if not model_output_dir.exists():
            model_output_dir.mkdir()
        nlp.to_disk(model_output_dir)
        print("Saved model to", model_out)



TRAIN_DATA = []
n_iter = 100
for input_file in input_files:

    if input_file.endswith('.xlsx'):
        get_train_data_from_excel(input_file)
        TRAIN_DATA.extend(get_train_data_from_excel(input_file))
    elif input_file.endswith('.json'):
        TRAIN_DATA.extend(get_train_data_from_json(input_file))
    else:
        print('BAD FILENAME')

model_train(TRAIN_DATA, n_iter,model_input_dir, model_output_dir)


