#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 07:38:39 2025

@author: cringwal
"""



import json 
import pandas as pd
import statistics

import wandb

api = wandb.Api()
API_KEY="YOUR API KEY"
wandb.login(key=API_KEY)
All_data=[]
####################" THIS FUNCTION ALLOWS TO 
def getHistoryMetrics(run):

    ###################### INIT EVERY VALUES
    #### F1 MACRO
    saturation_threshold=0.9
    F1_macro_saturated=False
    F1_macro_forgetting=0
    F1_macro_first_sat=None  
    F1_macro_broken_steps=[]   
    #### F1 MICRO
    F1_micro_saturated=False
    F1_micro_forgetting=0
    F1_micro_first_sat=None  
    F1_micro_broken_steps=[]  
    #### PARSING
    parsed_saturated=False
    parse_forgetting=0
    parsed_first_sat=None
    parsed_broken_steps=[]
    
    ################## GET THE COMPLETE HISTORIC FROM WANDB
    complete_hist={}
    #for i, row in run.history().iterrows():
    for row in run.scan_history():
        if str(row["epoch"])!="nan" and "val_loss" in row.keys() and str(row["val_loss"])!="nan":
            epo=str(row["epoch"]).replace(".0","")
            if(str(epo)!="None" ):  
                complete_hist[epo]={"val_loss":row["val_loss"],'val_part_parsed':row["val_part_parsed"],'val_F1_macro':row["val_F1_macro"],'val_F1_micro':row["val_F1_micro"]}
    
    ############# INIT VAL RELATED TO SAT
    #### PARSING
    last_val_part_parsed=None
    sat_val_part_parsed=None
    #### F1 MACRO
    last_val_F1_macro=None
    sat_val_F1_macro=None
    #### F1 MICRO
    last_val_F1_micro=None
    sat_val_F1_micro=None
    F1_RPT_sat=None
    ################################## ITERATE THE HISTORY
    for epoch in complete_hist.keys():
        print(epoch)
        nb_epoch=int(epoch)
        current_stats=complete_hist[epoch]
        
        ################################## PARSING
        if("val_part_parsed" in current_stats.keys()):
            val_part_parsed=current_stats["val_part_parsed"]
            ############## IF VALUE NOT NONE
            if(val_part_parsed and str(val_part_parsed)!="nan"):
                last_val_part_parsed=val_part_parsed
                ############ DIDN'T SATURATE FOR THE MOMENT
                if(parsed_saturated == False):                  
                    ######### SATURATION ? 
                    if(val_part_parsed>saturation_threshold):
                        parsed_saturated=True
                        parsed_first_sat=int(epoch)
                        sat_val_part_parsed=val_part_parsed
                        F1_RPT_sat=current_stats["val_F1_micro"]
                        print("SATURATE PARSING")
                        print("val_part_parsed: ",current_stats["val_part_parsed"])
                        print("val_F1+",current_stats["val_F1_micro"])
                ############ ALREADY SATURATE > BROKEN STEPS ? 
                elif(parsed_saturated and val_part_parsed < saturation_threshold ):
                    parsed_broken_steps.append(int(epoch))
            
        if("val_F1_macro" in current_stats.keys()):
            #print(current_stats)
            val_F1_macro=current_stats["val_F1_macro"]
            ############## IF VALUE NOT NONE
            if(val_F1_macro and str(val_F1_macro)!="nan"):
                last_val_F1_macro=val_F1_macro
                ############ DIDN'T SATURATE FOR THE MOMENT
                if(F1_macro_saturated == False):               
                    ######### SATURATION ?                 
                    if(val_F1_macro>saturation_threshold):
                        F1_macro_saturated=True
                        F1_macro_first_sat=int(epoch)
                        sat_val_F1_macro=val_F1_macro
                ############ ALREADY SATURATE > BROKEN STEPS ? 
                elif(F1_macro_saturated and val_F1_macro < saturation_threshold):
                    F1_macro_broken_steps.append(int(epoch))

        if("val_F1_micro" in current_stats.keys()):
            #print(current_stats)
            val_F1_micro=current_stats["val_F1_micro"]
            ############## IF VALUE NOT NONE
            if(val_F1_micro and str(val_F1_micro)!="nan"):
                last_val_F1_micro=val_F1_micro
                ############ DIDN'T SATURATE FOR THE MOMENT
                if(F1_micro_saturated == False):                
                    ######### SATURATION ?                 
                    if(val_F1_micro>saturation_threshold):
                        F1_micro_saturated=True
                        F1_micro_first_sat=int(epoch)
                        sat_val_F1_micro=val_F1_micro
                ############ ALREADY SATURATE > BROKEN STEPS ? 
                elif(F1_micro_saturated and val_F1_micro < saturation_threshold):
                    F1_micro_broken_steps.append(int(epoch))
                    
    ################## COMPUTE STABILITY
    ##### F1 MACRO
    #F1_macro_broken_ratio=None
    F1_macro_broken_ratio=0
    if(F1_macro_saturated):
        F1_macro_first_sat=int(F1_macro_first_sat)    
        if(saturation_threshold>last_val_F1_macro):
            F1_macro_forgetting=1
        #if(len(F1_macro_broken_steps)>0 ):
        F1_macro_broken_ratio=1-len(F1_macro_broken_steps)/(nb_epoch)
    ##### F1 MICRO
    #F1_micro_broken_ratio=None
    F1_micro_broken_ratio=0
    if(F1_micro_saturated):
        F1_micro_first_sat=int(F1_micro_first_sat)    
        if(saturation_threshold>last_val_F1_micro):
            F1_micro_forgetting=1
        #if(len(F1_micro_broken_steps)>0 ):
        F1_micro_broken_ratio=1-len(F1_micro_broken_steps)/(nb_epoch)
   
    
    ##### PARSING
    #parsed_broken_ratio=None
    parsed_broken_ratio=0
    if(parsed_saturated):  
       # print("parsed_first_sat>",parsed_first_sat)
        parsed_first_sat=int(parsed_first_sat)
           
        if(saturation_threshold>last_val_part_parsed):
            parse_forgetting=1
        #if(len(parsed_broken_steps)>0 ):
        parsed_broken_ratio=1-len(parsed_broken_steps)/(nb_epoch)

    ############# DID IT SATURATE DURING THE CURRENT FOLDS ? 
    parse_sat=None
    if(parsed_saturated):
        parse_sat=1
    F1_macro_sat=None
    if(F1_macro_saturated):
        F1_macro_sat=1
    F1_micro_sat=None
    if(F1_micro_saturated):
        F1_micro_sat=1
        
    return {"F1_RPT_sat":F1_RPT_sat,"F1_macro_sat":F1_macro_sat,"F1_macro_forgetting":F1_macro_forgetting,"F1_macro_first_sat":F1_macro_first_sat,"F1_macro_broken_ratio":F1_macro_broken_ratio,
    "F1_micro_sat":F1_micro_sat,"F1_micro_forgetting":F1_micro_forgetting,"F1_micro_first_sat":F1_micro_first_sat,"F1_micro_broken_ratio":F1_micro_broken_ratio,
    "parse_forgetting":parse_forgetting, "parsed_first_sat":parsed_first_sat,"parse_sat":parse_sat,"parsed_broken_ratio":parsed_broken_ratio}

#,"240ShadesOfSyntaxT5","CodeModels","OtherT5Models"
configs_dict={"T5_b_v_j":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_json-ld_T5_base_t5-base","B_b_v_g_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_tags_facto_BART_base_bart-base","B_b_v_u":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_TurtleUtlraLight_BART_bart-base","T5_b_v_u":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_TurtleUtlraLight_T5_t5-base","B_b_v_t_1_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleLight_1inLine_1facto_BART_base_bart-base","B_b_v_T":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtle_BART_base_bart-base","T5_b_v_t_1_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleS_1inLine_1facto_T5_base_t5-base","T5_b_v_x":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_xml_T5_base_t5-base","B_b_v_g":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_tags_BART_base_bart-base","T5_b_v_l":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_list_T5_base_t5-base","B_b_v_l_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_list_facto_BART_base_bart-base","B_b_v_l":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_list_BART_base_bart-base","B_b_v_t_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleLight_0inLine_1facto_BART_base_bart-base","T5_b_v_g_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_tags_facto_T5_base_t5-base","T5_b_v_t_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleS_0inLine_1facto_T5_base_t5-base","B_b_v_j":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_json-ld_BART_base_bart-base","B_b_v_x":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_xml_BART_base_bart-base","T5_b_v_t_1":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleS_1inLine_0facto_T5_base_t5-base","T5_b_v_l_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_list_facto_T5_base_t5-base","B_b_v_n":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_ntriples_BART_base_bart-base","T5_b_v_g":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_tags_T5_base_t5-base","T5_b_v_t":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleS_0inLine_0facto_T5_base_t5-base","B_b_v_t_1":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleLight_1inLine_0facto_BART_base_bart-base","B_b_v_t":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtleLight_0inLine_0facto_BART_base_bart-base","cT5_b_v_T":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/BASE_DS_turtle_codeT5_base_VOCAB512_codet5-base","cT5_b_v_t_1_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/BASE_DS_turtleS_0datatype_1inLine_1facto_codeT5_base_VOCAB2512_codet5-base","cT5_b_v_l":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_list_codet5_base512_codet5-base","cT5_b_v_t_1":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/BASE_DS_turtleS_0datatype_1inLine_0facto_codeT5_base_VOCAB2512_codet5-base","cT5_b_v_l_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_list_facto_codet5_base512_codet5-base","T5_b_v_T":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_turtle_T5_base_t5-base","cT5_b_v_t_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/BASE_DS_turtleS_0datatype_0inLine_1facto_codeT5_base_VOCAB2512_codet5-base","T5_b_v_n":"https://wandb.ai/celian-ringwald/12ShadesOfRDF/groups/DS_ntriples_T5_base_t5-base","cT5_b_v_t":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/BASE_DS_turtleS_0datatype_0inLine_0facto_codeT5_base_VOCAB2512_codet5-base","cT5_b_v_n":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/BASE_DS_ntriples_codeT5_base_VOCAB512_codet5-base","cT5_b_v_g":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_tags_codet5_base512_codet5-base","cT5_b_v_g_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_tags_facto_codet5_base512_codet5-base","cT5_b_v_u":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_TurtleUtlraLight_codeT5_codet5-base","cT5_b_v_j":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/BASE_DS_json-ld_codeT5_base_VOCAB512_codet5-base","pileT5_b_v_t_1":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_1inLine_0facto_PileT5_base_pile-t5-base","flanT5_b_v_l_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_list_facto_codet5_base512_codet5-base","flanT5_b_v_u":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_TurtleUtlraLight_flanT5_flan-t5-base","cT5_b_v_x":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_xml_T5_base_codeT5_base_VOCAB512_codet5-base","pileT5_b_v_n":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_ntriples_PileT5_base_pile-t5-base","pileT5_b_v_t_1_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_1inLine_1facto_PileT5_base_pile-t5-base","pileT5_b_v_u":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_TurtleUtlraLight_pileT5_pile-t5-base","pileT5_b_v_g":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_tags_PileT5_base_pile-t5-base","pileT5_b_v_l_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_list_facto_PileT5_base_pile-t5-base","flanT5_b_v_g":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_tags_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_t_1_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_1inLine_1facto_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_g_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_tags_facto_flant_tokenizer_OK_flan-t5-base","pileT5_b_v_t":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_0inLine_0facto_PileT5_base_pile-t5-base","flanT5_b_v_t_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_0inLine_1facto_flant_tokenizer_OK_flan-t5-base","pileT5_b_v_x":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_xml_PileT5_base_pile-t5-base","pileT5_b_v_t_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_0inLine_1facto_PileT5_base_pile-t5-base","pileT5_b_v_T":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtle_PileT5_base_pile-t5-base","pileT5_b_v_l":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_list_PileT5_base_pile-t5-base","pileT5_b_v_j":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_xml_PileT5_base_pile-t5-base","pileT5_b_v_g_f":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_tags_facto_PileT5_base_pile-t5-base","flanT5_b_v_x":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_xml_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_t_1":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_1inLine_0facto_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_T":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtle_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_t":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_turtleS_0datatype_0inLine_0facto_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_n":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_ntriples_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_l":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_list_flant_tokenizer_OK_flan-t5-base","flanT5_b_v_j":"https://wandb.ai/celian-ringwald/12ShadesOfRDFExtension/groups/DS_json-ld_flant_tokenizer_OK_flan-t5-base"}
for tag_name in configs_dict.keys():
#entity="celian-ringwald"
    model_uri=configs_dict[tag_name]
    
    print(model_uri)
    model_uri_split=model_uri.replace("https://wandb.ai/","").split("/")
    entity=model_uri_split[0]
    project=model_uri_split[1]
    group=model_uri_split[-1]
    print(entity,project,group)

#for c in configs_:
    #group=c["group"]
    #tag_name=c["tag_name"]
    best_score=0
    path_best=None
    print("================================")
    #if(results_data[project][k]["tag_name"]=="T5_b_v_j"):
    col_int=[
        "F1_micro_first_sat","F1_micro_broken_ratio",
        "F1_macro_first_sat","F1_macro_broken_ratio",
        "parsed_first_sat","parsed_broken_ratio",
        ]
    col_std=["test_F1_micro",
    "test_F1_macro"]
    col_means=["test_F1_micro",
    "test_F1_macro",
   # "test_prec_macro",
   # "test_recall_macro",
   # "test_prec_micro",
    # "test_recall_micro",
    #"test_dist_edit",
    #"test_BLEU",
    "test_part_parsed",
    "test_part_subj_ok",
    "test_part_valid",
    #"val_loss",
    #"carbon",
    #"nb_epoch",
    #"time",
    "F1_RPT_sat",
    "F1_micro_first_sat","F1_micro_broken_ratio",
    "F1_macro_first_sat","F1_macro_broken_ratio",
    "parsed_first_sat","parsed_broken_ratio"]
    #,"nb_epoch",
    
    col_sums=["parse_forgetting","parse_sat","F1_micro_forgetting","F1_micro_sat","F1_macro_forgetting","F1_macro_sat"]
    data_temp={}
    runs= api.runs(
        path=entity+"/"+project,
        filters={"$or": [{"group": group}]}
    )
    n_folds=0
    for run in runs:
        name=run.name
        print(name)
        n_folds+=1
        
        fold_tt="fold"+name.split("_")[-1]
        if fold_tt not in data_temp.keys():
            data_temp[fold_tt]={}
        data_temp[fold_tt]["nb_epoch"]=run.summary["epoch"]
        data_temp[fold_tt]["time"]=run.summary["_runtime"]/60
        
        for col in col_means:
            if(col not in data_temp[fold_tt].keys() or data_temp[fold_tt][col]==None):
                if(col in run.summary.keys()):
                    data_temp[fold_tt][col]= run.summary[col]
                    ##########################################"" FIX HERE THE FACT DIST ISN'T COMPUTED WHEN PRED=GOLD
                    if(col=="test_dist_edit" and str(run.summary[col]).lower()=="nan"):
                        data_temp[fold_tt][col]= 0
                else:
                    print(">>>>>>>>>>col missing : ",col)
                    data_temp[fold_tt][col]=None
                    break
        ############ HISTO METRICS
        histo_metrics=getHistoryMetrics(run)
        #print(histo_metrics)
        for m in histo_metrics.keys():
            data_temp[fold_tt][m]=histo_metrics[m]
        
        ##########################################"" FIX HERE THE FACT DIST ISN'T COMPUTED WHEN PRED=GOLD
        for f in data_temp.keys():
            if(f=="test_dist_edit" and str(run.summary[f]).lower()=="nan"):
                data_temp[f]["test_dist_edit"]= 0
            
    if(n_folds>0):
        data_temp2=[data_temp[k] for k in data_temp.keys()]
        data_aggr={}
        for col in col_int:
            inf=False
            list_val=[]
            for d in data_temp2:
               if col in d.keys():   
                   print(col)
                   if(d[col]!=None and str(d[col])!="NaN" and str(d[col])!="nan"):
                       if(isinstance(d[col], (int, float, complex))):
                           list_val.append(d[col])
                   else:
                       inf=True
        for col in col_std:
           vals=[]
           nb=0
           ok=True
           for d in data_temp2:
               if col in d.keys():
                   #print(col,":",d[col])
                   if(d[col]!=None and str(d[col])!="NaN" and str(d[col])!="nan"):
                       vals.append(d[col])
                       
           if(len(vals)==n_folds): 
               data_aggr[col+"_std"]=statistics.stdev(vals)
           else:
               data_aggr[col+"_std"]=None    
               
        for col in col_means:
           sum_=0
           nb=0
           ok=True
           for d in data_temp2:
               if col in d.keys():
                   #print(col,":",d[col])
                   if(d[col]!=None and str(d[col])!="NaN" and str(d[col])!="nan"):
                       sum_+=d[col]
                       nb+=1
                       
           if(nb==n_folds): 
               data_aggr[col]=sum_/nb
           else:
               data_aggr[col]=None
                
        for col in col_sums:
            sum_=0
            nb=0
            for d in data_temp2:
                if col in d.keys():
                    if(d[col]!=None and str(d[col])!="NaN" and str(d[col])!="nan"):
                        sum_+=d[col]
                        nb+=1
                    
            if(nb==n_folds): 
                data_aggr[col]=sum_
            else:
                data_aggr[col]=None
            
           # else:
            #    data_aggr[col]=None
        print("====================================")
        print(data_aggr)
        
        print("====================================")
        # and 'test_F1_macro' in data_aggr.keys()
        
        if(len(data_aggr.keys())>0 ):
          test_F1_micro=data_aggr['test_F1_micro']
          test_part_parsed=data_aggr['test_part_parsed']
          test_part_subj_ok=0
          test_part_valid=0
          if(int(test_part_parsed)>0):
              test_part_subj_ok=data_aggr['test_part_subj_ok']/test_part_parsed
              test_part_valid=data_aggr['test_part_valid']/test_part_parsed
          data_aggr['test_part_subj_ok']=test_part_subj_ok
          data_aggr['test_part_valid']=test_part_valid
          
          score_all=test_F1_micro*test_part_parsed*test_part_subj_ok*test_part_valid
          data_extended=data_aggr
          for col in list(set(col_means+col_sums)):
              if(col not in data_extended.keys()):
                  data_extended[col]=None
          data_extended["score"]=score_all
          data_extended["tag_name"]=tag_name
          All_data.append(data_extended)
        
    
df = pd.DataFrame.from_dict(All_data)
df2 = df.rename(columns={'test_part_parsed': 'R_TP', 'test_part_subj_ok': 'R_CS','test_part_valid':'R_SVT',
                        "test_F1_micro":"F1-","test_F1_macro":"F1+",
                        "test_prec_micro":"P-","test_prec_macro":"P+",
                        "test_recall_micro":"R-","test_recall_macro":"R+",
                        'test_BLEU':'B',"test_dist_edit":"lev",
                        'F1_micro_sat':"Sat_F1-","F1_micro_first_sat":"V_F1-","F1_micro_broken_ratio":"S_F1-",'F1_micro_forgetting':"D_F1-",
                        'F1_macro_sat':"Sat_F1+","F1_macro_first_sat":"V_F1+","F1_macro_broken_ratio":"S_F1+",'F1_macro_forgetting':"D_F1+",
                        'parse_sat':"Sat_RTP","parsed_first_sat":"V_RTP","parsed_broken_ratio":"S_RTP","parse_forgetting":"D_RTP",
                        'carbon':"Cc","time":"Tt","score":"Gg"
                        })
df2.to_csv("/user/cringwal/home/Desktop/New_RESULTS_SATURATION.csv")  
#df3 = df2[["tag_name","R_TP", "R_CS", "R_SVT","F1-","F1+","P-","P+","R-","R+","B","lev","nb_epoch",
#          "Sat_F1-","V_F1-","S_F1-","D_F1-","Sat_F1+","V_F1+","S_F1+","D_F1+",
#          "Sat_RTP","V_RTP","S_RTP","D_RTP","Cc","Tt","Gg","val_loss",'tokenizer_path', 'date', 'syntax', 'space', 'add_vocab','path_best', 
#        'model_name_or_path']]
#df3.sort_values('Gg', ascending=False).to_csv("//user/cringwal/home/Desktop/THESE/New_RESULTS_SATURATION.csv")  



