# TEXTO Ontology_matching

This repository contains our implementation of the ontology matching framework in the paper: **Ontology Matching using Textual Class Descriptions**

<h2 align="center">
  The framework of TEXTO
  <img align="center"  src="https://github.com/peng-yiwen/Ontolgy_matching/blob/main/TEXTO_overview.png" alt="...">
</h2>


### **Dependencies**
Dependencies can be installed using `requirements.txt`

### **Datasets**

- All the data used can be found in [data](https://github.com/peng-yiwen/Ontolgy_matching/tree/main/data) directory. The extended versions of the datasets are also available (denoted as yago-wikidata+, schema-wikidata+).
- The [data_script](https://github.com/peng-yiwen/Ontolgy_matching/tree/main/data_script) directory contains a Jupyter notebook detailing how we crawled the data, cleaned it, and converted it to OAEI standard format.

Dataset | #Classes | #Gold Mappings |
:--- | :---: | :---: 
NELL / DBpedia | 134/138 | 129 |
Wikidata / YAGO | 304/304 | 304 |
 Schema / Wikidata | 343/343 | 343 |

### How to run   
- To evaluate the TEXTO matching system on different datasets:**run directly main.ipynb file, remember to change the data_path to corresponding dataset**
