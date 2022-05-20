import tqdm
import json
import sys
import os
import math
import pandas as pd
import tempfile
import csv
import gzip
import xml.etree.cElementTree as ET
import glob
import random
from readers import pmreader2 as pmreader
from proposed_design_types import PROPOSED_DESIGN_TYPES


def abstract_plaintext(abstract):
    out = []
    for sec in abstract:
        if sec["header"] != "_UNSTRUCTURED":
            out.append(sec["header"])
        out.append(sec["text"])

    return "\n".join(out)

def iter_abstracts(filename, num_to_select = None):
    with open(filename, 'rb') as f:
        decompressedFile = gzip.GzipFile(fileobj=f, mode='rb')
        xml_dump = decompressedFile.read()
        xml_parsed = ET.fromstring(xml_dump)
        articles = xml_parsed.findall('PubmedArticle/MedlineCitation')
        if num_to_select:
            article_sample = random.sample(articles, num_to_select)
        else:
            article_sample = articles
        for j, article in enumerate(tqdm.tqdm(article_sample)):
            p_article = pmreader.PubmedCorpusReader(xml_ET=article)
            yield p_article.to_dict()
            
def is_proposed_design_type(ptyps):
    return set(ptyps).intersection(set(PROPOSED_DESIGN_TYPES)) != set()
            
def main():
    
    num_to_select = None

    filenames = glob.glob('/data/iain/pubmed/*.gz')
    
    pos_data = pd.DataFrame()


    num_todo = len(filenames)
    
    
    for i, fn in tqdm.tqdm(enumerate(filenames)): # this line does *all* >900 files (takes ages)
    # for id in range(len(filenames)):
    # for i, fn in tqdm.tqdm(enumerate([filenames[900]])): # file 900 looks to me to be a decent example; perhaps others better! but some earlier abstracts don't contain much info


            # here we're iterating through *all* files
            # in the final version, probably we'd want to randomly sample, to generate a manageable amount of data
            # I suspect we might not have enough RAM to hold all 300M articles at once, (and so you will need to sample in the loop)
            # but possibly it might work without this, and you could sample from the Pandas dataframe below instead!
            sys.__stdout__.write("file number {}/{} ({}) processing \n".format(i, num_todo, fn)) # optional to see progress from terminal
            print("file number {}/{} ({}) processing \n".format(i, num_todo, fn))
            data = []
            
            try:
                for article in iter_abstracts(fn, num_to_select):
                        data.append(article)

                data_pd = pd.DataFrame(data)

                pos_data_min = data_pd[data_pd['ptyp'].apply(is_proposed_design_type)]
                pos_data = pd.concat([pos_data, pos_data_min])

                pos_data.to_csv('pubmed_articles_pos.csv')
            except:
                pass

if __name__ == "__main__":
    main()