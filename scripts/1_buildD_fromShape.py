#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:20:07 2023

@author: cringwal
"""


import os
import sys

sys.path.append('/user/cringwal/home/Desktop/THESE/EXP_env/scripts/')
import json 

import triple_shapes as ts
import rdf_synthax_fct as rs
import abstract_reshape_fct as ar

from pyshacl import validate
from rdflib import Graph
import random

dir_path="/user/cringwal/home/Desktop/THESE/experiences/DBpedia_Person_lit_data/"
shape_file = "/user/cringwal/home/Desktop/THESE/EXP_env/scripts/PersonShape.ttl"

shacl_g = Graph()
shacl_g.parse(shape_file)

namespaces = shacl_g.namespaces()
prop_focus = ts.getShapeProp(shacl_g)
type_prop = ts.getShapePropWithType(shacl_g)
type_triples = ts.getShapeType(shacl_g)

saveResult=False
split_Result=False

files_prop=[]
files_abs=[]
# Iterate directory
for path in os.listdir(dir_path):
    if("entity_prop_dict" in path):
        files_prop.append(path)
    elif("abstract"):
        files_abs.append(path)
        
#### CREATE A DICT OF ENTITIES/ABSTRACT
ent_ab={}
for f in files_abs :
    with open(dir_path+f) as current_file:
        file_contents = json.load(current_file)
        for k in file_contents.keys():
            abstract=file_contents[k]
            ent_ab[rs.cleanEntURL(k)]=abstract
            

print(">>>>>>> NB entites : ",len(ent_ab.keys()))

################# CREATE A DICT ENTITES/PROP for entities having abstract
ent_ok=[]
ent_dict={}
prop_count_raw={p : 0 for p in prop_focus}
for f in files_prop :
    with open(dir_path+f) as current_file:
        file_contents = json.load(current_file)
        test=file_contents
        print(current_file)
        for k in file_contents.keys():
           if( rs.cleanEntURL(k) in ent_ab.keys()):
                valid=False
                for p in prop_focus:
                    if p in file_contents[k].keys():
                        valid=True
                    
                if(valid):
                    ent_ok.append(rs.cleanEntURL(k)) 
                    temp=[]
                    for p in prop_focus:
                        p2=p.replace("http://dbpedia.org/ontology/#","http://dbpedia.org/ontology/")
                        if(p in file_contents[k].keys()):
                            prop_count_raw[p]+=1
                            temp.append({"prop":p2,"value":file_contents[k][p][0]})
                        elif(p2 in file_contents[k].keys()):
                            prop_count_raw[p]+=1
                            temp.append({"prop":p2,"value":file_contents[k][p2][0]})
                    ent_dict[rs.cleanEntURL(k)]=temp

print(">>>>>>> NB entities in constraints scope : ",len(ent_dict.keys()))

############# FILTER ABSTRACT
abs_ok=[] 
for ent in ent_ok:
    abstract=ent_ab[ent]
    if( len(abstract)>5 and "{{" not in abstract and "}}" not in abstract):
        abs_ok.append(ent)
abs_ok=list(set(abs_ok))

print(">>>>>>> NB entities with abstract ok : ",len(abs_ok))
######### INIT RES DICT
dataset_focused={}
linear_types=["list","tags"]
for l in linear_types:
    dataset_focused[l]={}
RDF_types=["turtle","xml","json-ld","ntriples"]
for l in RDF_types:
    dataset_focused[l]={}

########## STATS
dict_not_found_prop={p:0  for p in prop_focus}
dict_found_prop={p:0  for p in prop_focus}
nb_ent_found=0
nb_triples_found=0
nb_ent_shapeOK=0

################## RANDOMIZE
abs_ok_random= sorted(abs_ok, key=lambda x: random.random())

    
Sample_limit=10000
SAMPLE_BART={}
SAMPLE_T5={}
################################ LOOK VALUES INTO ABSTRACT AND TRIPLES VIA SHACL
for i in range(len(abs_ok_random)):
    ent=abs_ok_random[i]
    print(">>>>>> ",i,"/",len(abs_ok_random),"/",nb_ent_shapeOK)
    if(nb_ent_shapeOK<Sample_limit):
        if(len(ent_dict[ent])>0):
            data_by_types={type_prop[k]:{} for k in type_prop.keys()}
            labels={}
            dates_found={}
            all_rel=[rel["prop"] for rel in ent_dict[ent]]
            for row in ent_dict[ent]:
                abstract=ent_ab[ent]
                prop=row["prop"]
                val=row["value"].replace('"'," ").strip()      
                type_p=type_prop[prop]
                if(val!="" and val.strip()!="" and val.strip()!=" "):
                    found=ts.find_in_abstract(abstract,prop,rs.cleanTxt(val),type_p)
                else:
                    found=False
               # print(prop,">",val," FOUND IN ABS ?",found)
    
                if(found==False and type_p == "xsd:string"):
                    special=ts.label_contains_special(val)
                    if(special==True):
                        val2=ts.simplify_label(val)
                        found2=ts.find_in_abstract(abstract,prop,val2,"xsd:string")
                        if(found2==True):
                            dict_found_prop[prop]+=1
                            if(prop not in labels.keys()):
                                data_by_types[type_p][prop]=[]
                            data_by_types["xsd:string"][prop].append(val2)
                        else:
                            dict_not_found_prop[prop]+=1
                elif(found==True):
                    
                    dict_found_prop[prop]+=1
                    if(prop not in data_by_types[type_p].keys()):
                        data_by_types[type_p][prop]=[]
                    data_by_types[type_p][prop].append(val)
                elif(found==False):
                    dict_not_found_prop[prop]+=1
       
            
            found_smth=[ True for k in data_by_types.keys() if len(list(data_by_types[k].keys()))]
            if(True in found_smth):   
                ent_dict2=[] 
                for k in data_by_types.keys():
                    for prop in data_by_types[k].keys():
                        temp_new={"type":k,"prop":prop,"value":data_by_types[k][prop]}
                        ent_dict2.append(temp_new)
                        
                if(len(ent_dict2)>0):
                    nb_ent_found+=1
                    nb_triples_found+=len(ent_dict2)
                    triples=ts.triplesWithShape(ent,ent_dict2,shacl_g)
                    
                    
                    ########## INFERENCE
                    triples=ts.DatesInferences(triples,"http://dbpedia.org/ontology/birthDate","http://dbpedia.org/ontology/birthYear")
                    triples=ts.DatesInferences(triples,"http://dbpedia.org/ontology/deathDate","http://dbpedia.org/ontology/deathYear")
                    
                    ############## VALIDATION
                    r = validate(triples, shacl_graph=shacl_g ,inference="rdfs")
                    conforms, results_graph, results_text = r
                    if(conforms==True):
                        nb_ent_shapeOK +=1
                        ################## SHORTEN ABSTRACT
                        ent3=ent.replace("http://dbpedia.org/resource/","").replace("https://dbpedia.org/resource/","")
                        
                        abstract_bart=ar.getShortenAbstract("bart-large", ent3, abstract)
                        abstract_t5=ar.getShortenAbstract("t5-large", ent3, abstract)
                        SAMPLE_BART[ent3]={"triples":triples.serialize(format="turtle"),"abstract":abstract_bart,"ent":ent3}
                        SAMPLE_T5[ent3]={"triples":triples.serialize(format="turtle"),"abstract":abstract_t5,"ent":ent3}
                        
                        #dataset_focused[rdf][ent]={"triples":triples.serialize(format=rdf),"abstract":abstract,"entity":ent}
                
                        # ent2=ent.replace("http://dbpedia.org/resource/","")
                        # t_list=ts.getTripleList(ent,ent_dict2,type_triples)
                        
                        
                        # dataset_focused["list"][ent]={"triples":t_list.replace("#",""),"abstract":abstract,"entity":ent2}
                        # t_tag=ts.getTripleWithTags(ent,ent_dict2,type_triples)
                        # dataset_focused["tags"][ent]={"triples":t_tag.replace("#",""),"abstract":abstract,"entity":ent2}
                        # for rdf in RDF_types:
                        #     dataset_focused[rdf][ent]={"triples":triples.serialize(format=rdf),"abstract":abstract,"entity":ent}
                    
    else:
        break


################# MAKE DIFF BART - T5 !!!!!!!!!!!!!!!!
if saveResult==True: 
    for model in ["BART","T5"]:
    
        ####### TURTLE 
        dataset_turtle=[]
        if(model == "BART"):
            SAMPLE = SAMPLE_BART
        elif(model == "T5"):
            SAMPLE = SAMPLE_T5
            
        for ent in SAMPLE.keys():        
            row=SAMPLE[ent]            
            dataset_turtle.append(row)
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_turtle.json", 'w', encoding='utf-8')  as f:
          json.dump(dataset_turtle, f)
       
    
    for model in ["BART","T5"]:
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_turtle.json",encoding="utf-8") as json_file:
               dataset_turtle = json.load(json_file) 
        print("CLEAN")
        dataset_turtle_2=[]
     # CLEAN TURTLE 
        for row in dataset_turtle:
             row["triples"]=rs.cleanTurtle(row["triples"])
             dataset_turtle_2.append(row)
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_turtle.json", 'w', encoding='utf-8')  as f:
             json.dump(dataset_turtle_2, f)
    # print("END")
    # with open("/user/cringwal/home/Desktop/THESE/experiences/exp3out/RAW_SAMPLE_data/DS_turtle.json", 'w', encoding='utf-8')  as f:
    #   json.dump(dataset_turtle_2, f)
    
    
    for model in ["BART","T5"]:
            
        if(model == "BART"):
            SAMPLE = SAMPLE_BART
        elif(model == "T5"):
            SAMPLE = SAMPLE_T5
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_turtle.json",encoding="utf-8") as json_file:
           dataset_turtle = json.load(json_file) 
        ###### LIST
        dataset_list_facto=[]
        dataset_list=[]
        for row in dataset_turtle:    
            triples= row["triples"]
            triples_list1=rs.TurtleToList(triples,True)
            new1=row.copy()
            new2=row.copy()
            # BE CAREFULL
            #triples_list1=str(triples_list1).replace("[","(").replace("]",")")
            new1["triples"]=str(triples_list1)
    
            dataset_list_facto.append(new1)
            
            triples_list2=rs.TurtleToList(triples,False)
            # BE CAREFULL
            #triples_list2=str(triples_list2).replace("[","(").replace("]",")")
            new2["triples"]=str(triples_list2)
            dataset_list.append(new2)
            
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_list.json", 'w', encoding='utf-8')  as f:
          json.dump(dataset_list, f)
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_list_facto.json", 'w', encoding='utf-8')  as f:
          json.dump(dataset_list_facto, f)
        ###### TAGS
        
        dataset_tags_facto=[]
        dataset_tags=[]
        for row in dataset_turtle:        
            triples= row["triples"]
            new1=row.copy()
            new2=row.copy()
            
            triples_list1=rs.TurtleToTag(triples,True)
            
            new1["triples"]=triples_list1
            dataset_tags_facto.append(new1)
            
            triples_list2=rs.TurtleToTag(triples,False)
            new2["triples"]=triples_list2
            dataset_tags.append(new2)
            
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_tags.json", 'w', encoding='utf-8')  as f:
            json.dump(dataset_tags, f)
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_tags_facto.json", 'w', encoding='utf-8')  as f:
            json.dump(dataset_tags_facto, f)
    
        ###### NTRIPLES
        dataset_ntriples=[]
        
        for row in dataset_turtle:        
            triples = row["triples"]
            new = row.copy()
            
            g1 = Graph()
            g1.parse(data=triples, format="turtle")
            
            ntrip = g1.serialize(format="ntriples")
            ntrip_correct = rs.reOrderNtriples(ntrip)
            
            new["triples"] = ntrip_correct
            dataset_ntriples.append(new)
            
        with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_ntriples.json", 'w', encoding='utf-8')  as f:
            json.dump(dataset_ntriples, f)
        
        ###### XML AND JSON-LD
        
        others_rdf=["xml","json-ld"]
        for current_format in others_rdf:
            print("CURRENT FORMAT >"+current_format)
            dataset_current=[]
            
            for row in dataset_turtle:     
                triples = row["triples"]
                new = row.copy()
                
                g1 = Graph()
                g1.parse(data=triples, format="turtle")
                
                current_translated = g1.serialize(format=current_format)
                
                new["triples"] = current_translated
                dataset_current.append(new)
                
            with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_"+current_format+".json", 'w', encoding='utf-8')  as f:
                json.dump(dataset_current, f)
   
        ###### TurtleS
        turtleS_conf=["0datatype_0inLine_0facto",
                      "1datatype_1inLine_1facto",
                      "1datatype_0inLine_0facto",
                      "0datatype_1inLine_0facto",
                      "0datatype_0inLine_1facto",
                      "0datatype_1inLine_1facto",
                      "1datatype_0inLine_1facto",
                      "1datatype_1inLine_0facto"
                      ]
        for conf in turtleS_conf:
           stripped_conf=conf.split("_")
           datatype=False
           inLine=False
           facto=False
           if(stripped_conf[0][0]=="1"):
               datatype=True
           if(stripped_conf[1][0]=="1"):
               inLine=True
           if(stripped_conf[2][0]=="1"):
               facto=True
           print("datatype>",datatype)
           print("inLine>",inLine)
           print("facto>",facto)
           dataset_turtle_current=[]
           
           for row in dataset_turtle: 
               
               triples= row["triples"]
               new = row.copy()
           
               triples_list1=rs.simplifyTutle(triples, datatype, inLine, facto)
               new["triples"] = triples_list1
               
               dataset_turtle_current.append(new)
           
           print(">>>>",conf)
           print("---------------------")
           print(">SAMPLE :")
           print(dataset_turtle_current[0]["triples"])
           # 
           print("SAVE IT")
           with open("/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data/DS_turtleS_"+conf+".json", 'w', encoding='utf-8')  as f:
                json.dump(dataset_turtle_current, f)
            
if split_Result==True:
    import  pandas as pd
    import numpy as np
    
    for model in ["BART","T5"]:
        raw_dir="/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/RAW_SAMPLE_data"
        dirpath="/user/cringwal/home/Desktop/THESE/experiences/exp4out/"+model+"/SPLITTED"
        files = os.listdir(raw_dir)
        for file_data in files :
            print(f)
            with open(raw_dir+"/"+file_data,encoding="utf-8") as json_file:
                data = json.load(json_file) 
            df = pd.DataFrame.from_dict(data, orient='columns') 
            train_size = 0.6
            validate_size = 0.2
            df = df.sample(frac = 1)
            train, validate, test = np.split(df.sample(frac=1), [int(train_size * len(df)), int((validate_size + train_size) * len(df))])
            f2=file_data.replace(".json","")
            with open(dirpath+"/"+f2+"_train.json", 'w', encoding='utf-8')  as fl:
                json.dump(train.to_dict(orient = 'records') , fl)
            with open(dirpath+"/"+f2+"_test.json", 'w', encoding='utf-8')  as fl:
                json.dump(test.to_dict(orient = 'records'), fl)
            with open(dirpath+"/"+f2+"_sample.json", 'w', encoding='utf-8')  as f1:
                  json.dump(test.to_dict(orient = 'records')[0:20], f1)
            with open(dirpath+"/"+f2+"_val.json", 'w', encoding='utf-8')  as fl:
                json.dump(validate.to_dict(orient = 'records'), fl)