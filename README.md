# 12ShadesOfRDFSyntax 

![QuestionIllustration](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/Screenshot%20from%202024-02-16%2016-49-51.png)

 :tada:The paper was accepted at ESWC 2024 at the Special Track on Large Language Models for Knowledge Engineering


If you use the code or cite our work, please reference this one as follows :
```
@InProceedings{10.1007/978-3-031-78952-6_8,
author="Ringwald, C{\'e}lian
and Gandon, Fabien
and Faron, Catherine
and Michel, Franck
and Akl, Hanna Abi",
editor="Mero{\~{n}}o Pe{\~{n}}uela, Albert
and Corcho, Oscar
and Groth, Paul
and Simperl, Elena
and Tamma, Valentina
and Nuzzolese, Andrea Giovanni
and Poveda-Villal{\'o}n, Maria
and Sabou, Marta
and Presutti, Valentina
and Celino, Irene
and Revenko, Artem
and Raad, Joe
and Sartini, Bruno
and Lisena, Pasquale",
title="12 Shades of RDF: Impact of Syntaxes on Data Extraction with Language Models",
booktitle="The Semantic Web: ESWC 2024 Satellite Events",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="81--91",
abstract="The fine-tuning of generative pre-trained language models (PLMs) on a new task can be impacted by the choice made for representing the inputs and outputs. This article focuses on the linearization process used to structure and represent, as output, facts extracted from text. On a restricted relation extraction (RE) task, we challenged T5 and BART by fine-tuning them on 12 linearizations, including RDF standard syntaxes and variations thereof. Our benchmark covers: the validity of the produced triples, the performance of the model, the training behaviours and the resources needed. We show these PLMs can learn some syntaxes more easily than others, and we identify a promising ``Turtle Light'' syntax supporting the quick and robust learning of the RE task.",
isbn="978-3-031-78952-6"
}



```
 # [UPDATE] Extended Benchmark 

We recently completed the benchmark by integrating three others models:

![QuestionIllustration](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/Models.png)


This material is based on a fork of [REBEL](https://github.com/Babelscape/rebel/tree/main)

But we also proposed a new Turtle syntax: [Turtle Ultra light](#35-turtle-ultra-light)

> All the finetuned model, results and training details are available here :
> * [https://wandb.ai/celian-ringwald/12ShadesOfRDF](https://wandb.ai/celian-ringwald/12ShadesOfRDF)
> * [https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension](https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension)

## Methodological framework

![Pipeline](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/120ShadesOfSyntaxes(2).drawio.png)

* We work on a [DBpedia Dump](https://auth.dbpedia.org/realms/dbpedia/protocol/openid-connect/auth?client_id=databus&scope=openid%20profile%20email&response_type=id_token&redirect_uri=https%3A%2F%2Fdatabus.dbpedia.org%2Fapp%2Fcallback&response_mode=form_post&prompt=none&nonce=yBd4sV9lC_xTEG9XV_rX-ndo4btioIuXOp03UNBaIKc&state=eyJyZXR1cm5UbyI6Ii9kYnBlZGlhL2NvbGxlY3Rpb25zL2RicGVkaWEtc25hcHNob3QtMjAyMi0wOSIsImF0dGVtcHRpbmdTaWxlbnRMb2dpbiI6dHJ1ZX0), we load into a local [CORESE triple store](https://github.com/Wimmics/corese), we then extract only datatypes triples of dbo:Person in JSON.
* Code of steps (1.1.) (1.2.), (2.1.), (2.2.) and (3) can be found in [1_buildD_fromShape.py](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/scripts/1_buildD_fromShape.py)
* K-fold validation is based on https://github.com/SkafteNicki/pl_crossvalidate/tree/master
* As usual in a Pytorch lightning project our configuration is divided into data/train/model directory
    * BART : [training config](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/conf/train/bart_dbpedia.yaml) - [model config](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/conf/model/bart_base_model.yaml)
    * T5 : [training config](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/conf/train/t5_dbpedia.yaml) - [model config](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/conf/model/t5_base_model.yaml)
    * Each configuration related to syntaxes can be found at https://github.com/datalogism/12ShadesOfRDFSyntax/tree/main/conf/data
 * Specific parser and triples utils: https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/src/triples_fct.py 
 * Metrics computation implementation: https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/src/score_fct.py, https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/src/score.py

[EDIT]: We better formalise our training behaviors metrics for more details check [this page](https://github.com/datalogism/12ShadesOfRDFSyntax/tree/main/eval)
[EDIT]: We recently proposed a new script to automatise the computation of the meta-metatrics from wandb

## SHACL Shape used
```
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dbo: <http://dbpedia.org/ontology/> .


schema:PersonShape a sh:NodeShape ;
 sh:targetClass dbo:Person ; 
 sh:property [ 
   sh:path rdfs:label;
   sh:minCount 1 ;
   sh:datatype xsd:string;
 ];
sh:or (
    [
      sh:property [ 
         sh:path dbo:birthDate;
         sh:datatype xsd:date;
         sh:minCount 1;
         sh:maxCount 1;
      ]
    ]   
    [
      sh:property [ 
          sh:path dbo:birthYear;
          sh:datatype xsd:gYear;
          sh:minCount 1;
          sh:maxCount 1;
      ]
    ]
);
sh:property [ 
   sh:path dbo:deathYear;
   sh:minCount 0;
   sh:maxCount 1;
   sh:datatype xsd:gYear;
 ];
 sh:property [ 
   sh:path dbo:alias;
   sh:datatype xsd:string ;
   sh:minCount 0;
   sh:maxCount 10;
   sh:nodeKind sh:Literal;
 ];
 sh:property [ 
   sh:path dbo:birthName ;
   sh:datatype xsd:string ;
   sh:minCount 0;
   sh:maxCount 1;
   sh:nodeKind sh:Literal ;
 ] ;
 sh:property [ 
   sh:path dbo:deathDate ;
   sh:datatype xsd:date ;
   sh:minCount 0;
   sh:maxCount 1;
 ].
```

## Syntax details

### 1. Syntax of literature
#### 1.1. LIST
* Normal:
```
[['Homer_Simpson', 'type', 'Person'], ['Homer_Simpson', 'label', 'Homer Simpson'], ['Homer_Simpson', 'birthDate', '1987-04-19'], ['Homer_Simpson', 'birthYear', '1987']]
```
* Factorised:
```
[['Homer_Simpson',[ ['type', 'Person'], ['label', 'Homer Simpson'], ['birthDate', '1987-04-19'], ['birthYear', '1987']]]
```
* ADDED vocab:
```["[","]",",","'"]```
#### 1.2. TAGS
* Normal:
```
<subj>Homer_Simpson<rel>type<obj>Person<et><subj>Homer_Simpson<rel>label<obj>Homer Simpson<et><subj>Homer_Simpson<rel>birthDate<obj>1987-04-19<et><subj>Homer_Simpson<rel>birthYear<obj>1987<et>
```
* Factorized:
```
<subj>Homer_Simpson<rel>type<obj>Person<et><rel>label<obj>Homer Simpson<et><rel>birthDate<obj>1987-04-19<et><rel>birthYear<obj>1987<et>
```
* ADDED vocab:
```["<subj>","<rel>","<obj>","<et>"]```

### 2. RDF syntaxes
#### 2.1. Turtle
```
@prefix dbo: <http://dbpedia.org/ontology/>.
@prefix dbr: <http://dbpedia.org/resource/>.
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.

dbr:Homer_Simpson a dbo:Person ;
    rdfs:label "Homer Simpson"^^<xsd:string> ;
    dbo:birthDate "1987-04-19"^^<xsd:date> ;
    dbo:birthYear "1987"^^<xsd:gYear>.
```

* ADDED vocab:
```
# FOLLOWING  https://www.w3.org/TR/rdf12-turtle/#language-features
[".",",",";","\n",":","<",">"]
# IRI ref
["@base","@prefix"]
# Literals
["'",'"',"'''",'"""']
# Languages and datatype
["^^","@"]
# List / SET / TRIG
["[","]","{","}","(",")"]
# blanck nodes + a > rdf:type
["_:"," a "]
```
#### 2.2. XML
```
<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF
   xmlns:dbo="http://dbpedia.org/ontology/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
>
  <rdf:Description rdf:about="http://dbpedia.org/resource/Homer_Simpson">
    <rdf:type rdf:resource="http://dbpedia.org/ontology/Person"/>
    <dbo:birthDate rdf:datatype="xsd:date">1987-04-19</dbo:birthDate>
    <rdfs:label rdf:datatype="xsd:string">Homer Simpson</rdfs:label>
    <dbo:birthYear rdf:datatype="xsd:gYear">1987</dbo:birthYear>
  </rdf:Description>
</rdf:RDF>
```
* ADDED VOCAB:
```
FROM #https://www.w3.org/TR/rdf-syntax-grammar/
# "/>" not added could broke URI
syntax_vocab=syntax_vocab+[":","</","<",">"," />","\n","="]
# Head and comment
syntax_vocab=syntax_vocab+["<?","?>","<!--","--!>"]
# CORE SYNTHAX TERM
syntax_vocab=syntax_vocab+["rdf:RDF","rdf:ID","rdf:type","rdf:about","rdf:parseType","rdf:resource","rdf:nodeID","rdf:datatype"]
# LIST
syntax_vocab=syntax_vocab+["rdf:li","rdf:_"]
# XML 
syntax_vocab=syntax_vocab+["xml:lang","xml:base","xmlns:"]
```

#### 2.3. JSON LD
```
[
 {
  "@id": "http://dbpedia.org/resource/Homer_Simpson",
  "@type": [
  "http://dbpedia.org/ontology/Person"
  ],
  "http://dbpedia.org/ontology/birthDate": [
  {
    "@type": "xsd:date",
    "@value": "1987-04-19"
    }
                              "  ],
  "http://dbpedia.org/ontology/birthYear": [
  {
    "@type": "xsd:gYear",
    "@value": "1987"
    }
  ],
  "http://www.w3.org/2000/01/rdf-schema#label": [
  {
    "@type": "xsd:string",
    "@value": "Homer Simpson"
    }
  ]
 }
]


[
 {
  "@id": "http://dbpedia.org/resource/Homer_Simpson",
  "@type": [
   "http://dbpedia.org/ontology/Person"
   ],
    "http://dbpedia.org/ontology/birthDate": [
        {
          "@type": "xsd:date",
          "@value": "1987-04-19"
        }
      ],
      "http://dbpedia.org/ontology/birthYear": [
        {
          "@type": "xsd:gYear",
          "@value": "1987"
        }
      ],
      "http://www.w3.org/2000/01/rdf-schema#label": [
        {
          "@type": "xsd:string",
          "@value": "Homer Simpson"
        }
      ]
  }
]
```
* ADDED VOCAB:
```
# JSON STANDARDS
#DELETED NOT FOUND IN JSON "'"
# FROM https://ecma-international.org/publications-and-standards/standards/ecma-404/
# we kept \n
syntax_vocab=syntax_vocab+["[","]","{","}",":",",",'"',"\n"]
# W3C JSON-LD standard
# https://www.w3.org/TR/json-ld11/#terms
#term
syntax_vocab=syntax_vocab+["@type"]
# node objects
syntax_vocab=syntax_vocab+["@set","@list","@value","@context","@id","@included","@graph","@nest","@reverse","@index"]
# frame objects
syntax_vocab=syntax_vocab+["@default","@null","@none","@embed","@always","@once","@explicit","@default","@omitDefault","@requiereAll"]
# values objects
syntax_vocab=syntax_vocab+["@value","@language","@direction"]
# propery_based Index Maps        
syntax_vocab=syntax_vocab+["@container"]
# included block
syntax_vocab=syntax_vocab+["@included"]        
# context def
syntax_vocab=syntax_vocab+["@import","@base","@propagate","@protected","@version","@vocab"]    
# other
syntax_vocab=syntax_vocab+["@json","@prefix"]
```
#### 2.4. NTRIPLES 
```
<http://dbpedia.org/resource/Homer_Simpson> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person> 
<http://dbpedia.org/resource/Homer_Simpson> <http://dbpedia.org/ontology/birthYear> "1987"^^<xsd:gYear> .
<http://dbpedia.org/resource/Homer_Simpson> <http://dbpedia.org/ontology/birthDate> "1987-04-19"^^<xsd:date> .
<http://dbpedia.org/resource/Homer_Simpson> <http://www.w3.org/2000/01/rdf-schema#label> "Homer Simpson"^^<xsd:string> ..
```
* ADDED VOCAB:

 ```
# FOLLOWING https://www.w3.org/TR/n-triples/
syntax_vocab=syntax_vocab+["<",">",".","\n"]
# quote
syntax_vocab=syntax_vocab+['"']
# datatypes 
syntax_vocab=syntax_vocab+["^^","@"]
# blank nodes
syntax_vocab=syntax_vocab+["_:"]
 ```

### 3. Syntaxes proposed : **Turtle Light**
* ADDED VOCAB
```
syntax_vocab=syntax_vocab+[".",",",";","\n",":","<",">"]
 # Literals
 syntax_vocab=syntax_vocab+['"']
 # Languages and datatype
 if(conf.datatype==True):
     syntax_vocab=syntax_vocab+["^^","@"]
 # List / SET / TRIG
 syntax_vocab=syntax_vocab+["[","]","{","}","(",")"]
 # blanck nodes + a > rdf:type
 syntax_vocab=syntax_vocab+["_:"," a "]
 ```
#### 3.1. Turtle Light factorised / multilines
```
:Homer_Simpson a :Person ;
   :label "Homer Simpson" ;
   :birthDate "1987-04-19" ;
   :birthYear "1987".
```
#### 3.2. Turtle Light factorised / in-line
```
:Homer_Simpson a :Person ; :label "Homer Simpson" ; :birthDate "1987-04-19" ; :birthYear "1987".
```
#### 3.3. Turtle Light not factorised / multilines
```
:Homer_Simpson a :Person ;
:Homer_Simpson :label "Homer Simpson" ;
:Homer_Simpson :birthDate "1987-04-19" ;
:Homer_Simpson :birthYear "1987".
```
#### 3.4. Turtle Light not factorised / in-line
```
:Homer_Simpson a :Person ; :Homer_Simpson :label "Homer Simpson" ; :Homer_Simpson :birthDate "1987-04-19" ; :Homer_Simpson :birthYear "1987".
```

#### 3.5. Turtle Ultra Light
```
Homer_Simpson a Person ; label "Homer Simpson" ; birthDate "1987-04-19" ; birthYear "1987".
```

## RUNNING EXPERIMENTS

* Changing path files
### TRAIN
```
   python ./src/train_withShape.py model=t5_base data=Azzzura_DS_turtleS_0datatype_1inLine_1facto_t5 train=t5_dbpedia 
```
### TEST
```
   python ./src/test_withShape.py model=bart_base_model data=Azzzura_DS_turtle_bart train=bart_dbpedia checkpoint_path=$checkpoint_path tokenizer_path=$tokenizer_path
```

Being able to test out of the training pipeline and solve [https://github.com/Babelscape/rebel/issues/55](https://github.com/Babelscape/rebel/issues/55), we extended [pl_modules.py](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/src/pl_modules.py) by redefining checkpoint loaders classes.
