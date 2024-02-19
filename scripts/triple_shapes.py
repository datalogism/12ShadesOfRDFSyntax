#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:51:04 2023

@author: cringwal
"""

import itertools
import datefinder
from rdflib import Graph, URIRef, Literal, BNode,Namespace
from rdflib.namespace import RDF
from unidecode import unidecode
import random
import urllib

def getShapeType(shacl_g):
    get_types = """
        SELECT DISTINCT ?target_class
        WHERE {
            ?a sh:targetClass ?target_class
        }"""
    qres = shacl_g.query(get_types)
    return [ str(row[0])for row in qres][0]

def getShapeProp(shacl_g):
    
    get_prop = """
    SELECT DISTINCT ?target_prop
    WHERE {
        ?a sh:path ?target_prop
    }"""
    qres = shacl_g.query(get_prop)
    return [ str(row[0]) for row in qres]

def getShapePropWithType(shacl_g):
    get_prop = """
    SELECT DISTINCT ?target_prop ?datatype
    WHERE {
        ?a sh:path ?target_prop;
           sh:datatype ?datatype.
    }"""
    qres = shacl_g.query(get_prop)
    return {  str(row[0]) : str(row[1]).replace("http://www.w3.org/2001/XMLSchema#","xsd:") for row in qres}


           
def find_in_abstract(abstract,prop,value,type_prop):
    if(type_prop=="xsd:gYear"):
        matches = datefinder.find_dates(abstract)
        dates=[]
        try:
            for match in matches:
                if(match!=''):
                    dates.append(match.strftime('%Y'))
        except Exception as error:
            print(error)
            pass
        if(len(dates)>0):
            if(value in dates):
                return True
            else: 
                return False
    if(type_prop=="xsd:date"):
        matches = datefinder.find_dates(abstract)
        dates=[]
        try:
            for match in matches:
                if(match!=''):
                    dates.append(match.strftime('%Y-%m-%d'))
        except Exception as error:
            print(error)
            pass
        if(len(dates)>0):
            if(value in dates):
                return True
            else: 
                return False
    elif(type_prop=="xsd:string"):
        if(value.lower() in abstract.lower()):
            return True
        else :
           if(unidecode(value.lower()) in unidecode(abstract.lower())):
               return True 
    return False

#def clean_abstract(abstract):
#https://en.wikipedia.org/wiki/Category:Wikipedia_naming_conventions
def label_contains_special(label):
    if("(" in label or "." in label or "," in label):
        return True
    else:
        return False
def simplify_label(label):
    temp_label=label
    if("(" in temp_label):
        temp_label=label[:label.index("(")]
    if("," in temp_label):        
        temp_label=label[:label.index(",")]
    
    if("." in temp_label):
        temp_label2 =temp_label.split()
        temp_label_clean = []
        for token in temp_label2:
            if("." not in token):
                temp_label_clean.append(token)
        temp_label=" ".join(temp_label_clean)
    return temp_label.strip()

def replace_ns_entity(entity_k,uri_pattern):
    return entity_k.replace("http://dbpedia.org/resource/",uri_pattern)

def triplesWithShape(ent_k,list_relations,shacl_g):
    type_triple=getShapeType(shacl_g)
    names_spaces=shacl_g.namespaces()
    
    g = Graph()
    dbr = Namespace("http://dbpedia.org/resource/")
    g.bind("dbr", dbr)
    for ns_prefix, namespace in names_spaces:
        current_ns=Namespace(str(namespace))
        g.bind(ns_prefix, current_ns)
    
    type_triple_uri = URIRef(type_triple)
    current_entity = URIRef(ent_k)
    
    g.add((current_entity,URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),type_triple_uri))
    for rel in list_relations:       
        prop_uri=URIRef(rel["prop"])
        for v in rel["value"]:
            obj_val=Literal(v, datatype=rel["type"].replace("xsd:","http://www.w3.org/2001/XMLSchema#"))
            g.add((current_entity,prop_uri,obj_val))
        
    return g

def triples(ent_k,list_relations,type_triple):
    dbo = Namespace("http://dbpedia.org/ontology/")
    dbr = Namespace("http://dbpedia.org/resource/")
    rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    xsd=Namespace("http://www.w3.org/2001/XMLSchema#")
    person = URIRef("http://dbpedia.org/ontology/Person")
    g = Graph()
    g.bind("dbo", dbo)
    g.bind("dbr", dbr)
    g.bind("rdf", rdf)
    g.bind("rdfs", rdfs)
    g.bind("xsd", xsd)
    current_entity = URIRef(ent_k)
    
    g.add((current_entity,URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),person))
    for rel in list_relations:       
        prop_uri=URIRef(rel["prop"])
        obj_val=Literal(rel["value"],datatype=rel["type"])
        g.add((current_entity,prop_uri,obj_val))
    
    return g
  
def getTripleList(ent_k,list_relations,type_ent):
    ent_simple=ent_k.replace("http://dbpedia.org/resource/#","").replace("http://dbpedia.org/resource/","")
    type_simple=type_ent.replace("http://dbpedia.org/ontology/#","").replace("http://dbpedia.org/ontology/","")
    list_=[[ent_simple,"type",type_simple]]
    for rel in list_relations:
        rel2=rel["prop"].replace("http://www.w3.org/2000/01/rdf-schema#","").replace("http://dbpedia.org/ontology/","")
        for v in rel["value"]:
            list_.append([ent_simple,rel2,v])
    #str_l=str(list_).replace("[","(").replace("]",")")
    return list_

def getTripleWithTags(ent_k,list_relations,type_ent):
    ent_simple=ent_k.replace("http://dbpedia.org/resource/#","").replace("http://dbpedia.org/resource/","")
    type_simple=type_ent.replace("http://dbpedia.org/ontology/#","").replace("http://dbpedia.org/ontology/","")
    list_="<subj>"+ent_simple+"<rel>type<obj>"+type_simple+"<et>"
    for rel in list_relations:
        rel2=rel["prop"].replace("http://www.w3.org/2000/01/rdf-schema#","").replace("http://dbpedia.org/ontology/","")
        for v in rel["value"]:
            list_=list_+"<subj>"+ent_simple+"<rel>"+rel2+"<obj>"+v+"<et>"
    return list_


def get_RDFtriples(ent_k,list_relations,syntax):
    dbo = Namespace("http://dbpedia.org/ontology/")
    dbr = Namespace("http://dbpedia.org/resource/")
    rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    xsd=Namespace("http://www.w3.org/2000/01/rdf-schema#")
    person = URIRef("http://dbpedia.org/ontology/Person")
    g = Graph()
    g.bind("dbo", dbo)
    g.bind("dbr", dbr)
    g.bind("rdf", rdf)
    g.bind("rdfs", rdfs)
    current_entity = URIRef(ent_k)
    
    g.add((current_entity, RDF.type,person))
    for rel in list_relations:
        
        prop_uri=URIRef(rel["prop"])
        obj_val=Literal(rel["value"])
        g.add((current_entity,prop_uri,obj_val))
    
    
    s=g.serialize(format=syntax)
    s2=s.replace("\n\n","\n").split("\n")
    list_s2=[]
    for tr in s2:
        if("prefix" not in tr and tr!=""):
            list_s2.append(tr)
    #random.shuffle(list_s2)
    s3="\n".join(list_s2)
    #s4="<"+ent_k+"> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person> .\n"+s3+' .\n\n'
    return s3


def DatesInferences(triples,date_prop,year_prop):
    get_prop = """SELECT ?s ?date WHERE {
        ?s <"""+date_prop+"""> ?date.
        FILTER NOT EXISTS { ?s <"""+year_prop+"""> ?year }
        }
    """
    qres = triples.query(get_prop)
    results=[q for q in qres]
    if(len(list(qres))>0):
       ent=str(results[0][0])
       val=str(results[0][1]).replace("['","").replace("']","")[0:4]
       triples.update("INSERT DATA { <"+ent+"> <"+year_prop+"> '"+val+"'^^xsd:gYear }")
    return triples

