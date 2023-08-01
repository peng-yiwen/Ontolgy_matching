from owlready2 import *
from rdflib import Namespace, URIRef
import csv


def rdf2csv(input_file_path, output_file):
    onto = World()
    onto.get_ontology(input_file_path).load()

    with open(output_file, 'w', newline='') as f_out:
        colums = ['id', 'label', 'comment']
        writer = csv.DictWriter(f_out, fieldnames=colums)
        writer.writeheader()

        for cls in onto.classes():
            cls_id = cls.iri
            cls_label = cls.label.first() if cls.label else None
            cls_comment = cls.comment.first() if cls.comment else None
            writer.writerow({'id': cls_id, 'label': cls_label, 'comment': cls_comment})


def reference2csv(input_file_path, output_file):
    onto = World()
    onto.get_ontology(input_file_path).load()
    g = onto.as_rdflib_graph()

    namespace = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment")
    Entity1 = URIRef(namespace + 'entity1')
    Entity2 = URIRef(namespace + 'entity2')
    Relation = URIRef(namespace + 'relation')

    with open(output_file, 'w', newline='') as f_out:
        colums = ['Class1_id', 'Class2_id', 'Relation']
        writer = csv.DictWriter(f_out, fieldnames=colums)
        writer.writeheader()

        for s, p, o in g:
            if p == Relation and str(o) == '=':
                entity1 = [str(o) for s, p, o in g.triples((s, Entity1, None))][0]
                entity2 = [str(o) for s, p, o in g.triples((s, Entity2, None))][0]
                writer.writerow({'Class1_id': entity1, 'Class2_id': entity2, 'Relation': '='})