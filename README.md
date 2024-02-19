# 12ShadesOfRDFSyntax

![QuestionIllustration](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/Screenshot%20from%202024-02-16%2016-49-51.png)

This material is based on a fork of [REBEL](https://github.com/Babelscape/rebel/tree/main)



## Syntax examples

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
#### 1.2. TAGS
* Normal:
```
<subj>Homer_Simpson<rel>type<obj>Person<et><subj>Homer_Simpson<rel>label<obj>Homer Simpson<et><subj>Homer_Simpson<rel>birthDate<obj>1987-04-19<et><subj>Homer_Simpson<rel>birthYear<obj>1987<et>
```
* Factorized:
```
<subj>Homer_Simpson<rel>type<obj>Person<et><rel>label<obj>Homer Simpson<et><rel>birthDate<obj>1987-04-19<et><rel>birthYear<obj>1987<et>
```
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
#### 2.4. NTRIPLES 
```
<http://dbpedia.org/resource/Homer_Simpson> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person> 
<http://dbpedia.org/resource/Homer_Simpson> <http://dbpedia.org/ontology/birthYear> "1987"^^<xsd:gYear> .
<http://dbpedia.org/resource/Homer_Simpson> <http://dbpedia.org/ontology/birthDate> "1987-04-19"^^<xsd:date> .
<http://dbpedia.org/resource/Homer_Simpson> <http://www.w3.org/2000/01/rdf-schema#label> "Homer Simpson"^^<xsd:string> ..
```
### 3. Syntaxes proposed : **Turtle Light**

#### 3.1. Turtle Light factorised / multilines
```
:Homer_Simpson a :Person ;
   :label "Homer Simpson" ;
   :birthDate "1987-04-19" ;
   :birthYear "1987".
```
#### 3.1. Turtle Light factorised / in-line
```
:Homer_Simpson a :Person ; :label "Homer Simpson" ; :birthDate "1987-04-19" ; :birthYear "1987".
```
#### 3.1. Turtle Light not factorised / multilines
```
:Homer_Simpson a :Person ;
:Homer_Simpson :label "Homer Simpson" ;
:Homer_Simpson :birthDate "1987-04-19" ;
:Homer_Simpson :birthYear "1987".
```
#### 3.1. Turtle Light not factorised / in-line
```
:Homer_Simpson a :Person ; :Homer_Simpson :label "Homer Simpson" ; :Homer_Simpson :birthDate "1987-04-19" ; :Homer_Simpson :birthYear "1987".
```
## Models

## Added Vocabulary

## Metrics
