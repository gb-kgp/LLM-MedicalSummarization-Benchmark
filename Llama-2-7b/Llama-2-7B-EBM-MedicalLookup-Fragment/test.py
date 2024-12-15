from Bio import Entrez
import sys
import csv

Entrez.email = 'enter.email@gmail.com'
 
def fetch_abstracts(pub_ids, retmax=1000, output_file='abstracts.csv'):    
    # Make sure requests to NCBI are not too big
    for i in range(0, len(pub_ids), retmax):
        j = i + retmax
        if j >= len(pub_ids):
            j = len(pub_ids)

        print(f"Fetching abstracts from {i} to {j}.")
        handle = Entrez.efetch(db="pubmed", id=','.join(pub_ids[i:j]),
                        rettype="xml", retmode="text", retmax=retmax)
        
        records = Entrez.read(handle)

        abstracts = [pubmed_article['MedlineCitation']['Article']['ArticleTitle']+ '\n' +
                     pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                     for pubmed_article in records['PubmedArticle']]

        abstract_dict = dict(zip(pub_ids[i:j], abstracts))

        with open(output_file, 'a', newline='') as csvfile:
            fieldnames = ['pub_id', 'abstract']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
            if i == 0:
              writer.writeheader()
            for pub_id, abstract in abstract_dict.items():
              writer.writerow({'pub_id': pub_id, 'abstract': abstract})

if __name__ == '__main__':
  filename = sys.argv[1]
  pub_ids = open(filename, "r").read().splitlines()
  fetch_abstracts(pub_ids)
