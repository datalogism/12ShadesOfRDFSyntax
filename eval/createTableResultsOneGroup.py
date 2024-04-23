#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:24:48 2024

@author: cringwal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:38:24 2024

@author: cringwal
"""

import json 
import pandas as pd


import wandb

api = wandb.Api()
API_KEY="#############################"
wandb.login(key=API_KEY)

def getHistoryMetrics(run):
    
    F1_macro_saturated=False
    F1_macro_forgetting=0
    F1_macro_first_sat=None  
    F1_macro_broken_steps=[]   

    F1_micro_saturated=False
    F1_micro_forgetting=0
    F1_micro_first_sat=None  
    F1_micro_broken_steps=[]  
    
    
    parsed_saturated=False
    parse_forgetting=0
    parsed_first_sat=None
    parsed_broken_steps=[]
    
    complete_hist={}
    for i, row in run.history().iterrows():
        if str(row["epoch"])!="nan" and "val_loss" in row.keys() and str(row["val_loss"])!="nan":
            epo=str(row["epoch"]).replace(".0","")
            complete_hist[epo]={"val_loss":row["val_loss"],'val_part_parsed':row["val_part_parsed"],'val_F1_macro':row["val_F1_macro"],'val_F1_micro':row["val_F1_micro"]}
    
    last_val_part_parsed=None
    sat_val_part_parsed=None

    last_val_F1_macro=None
    sat_val_F1_macro=None

    last_val_F1_micro=None
    sat_val_F1_micro=None

    for epoch in complete_hist.keys():
        nb_epoch=int(epoch)
        current_stats=complete_hist[epoch]
        if("val_part_parsed" in current_stats.keys()):
            #print(current_stats)
            val_part_parsed=current_stats["val_part_parsed"]
            #print(val_part_parsed)
            if(str(val_part_parsed)!="nan"):
                last_val_part_parsed=val_part_parsed
                if(parsed_saturated == False):                    
                    if(val_part_parsed>0.9):
                        parsed_saturated=True
                        parsed_first_sat=int(epoch)
                        sat_val_part_parsed=val_part_parsed
                elif(parsed_saturated and val_part_parsed < sat_val_part_parsed ):
                    parsed_broken_steps.append(int(epoch))
            
        if("val_F1_macro" in current_stats.keys()):
            #print(current_stats)
            val_F1_macro=current_stats["val_F1_macro"]
            if(str(val_F1_macro)!="nan"):
                last_val_F1_macro=val_F1_macro
                if(F1_macro_saturated == False):                    
                    if(val_F1_macro>90):
                        print("F1_macro_saturated!!!")
                        F1_macro_saturated=True
                        F1_macro_first_sat=int(epoch)
                        sat_val_F1_macro=val_F1_macro
                elif(F1_macro_saturated and val_F1_macro < sat_val_F1_macro):
                    F1_macro_broken_steps.append(int(epoch))

        if("val_F1_micro" in current_stats.keys()):
            #print(current_stats)
            val_F1_micro=current_stats["val_F1_micro"]
            if(str(val_F1_micro)!="nan"):
                last_val_F1_micro=val_F1_micro
                if(F1_micro_saturated == False):                    
                    if(val_F1_micro>90):
                        print("F1_micro_saturated!!!")
                        F1_micro_saturated=True
                        F1_micro_first_sat=int(epoch)
                        sat_val_F1_micro=val_F1_micro
                elif(F1_micro_saturated and val_F1_micro < sat_val_F1_micro):
                    F1_micro_broken_steps.append(int(epoch))
                    
    print("================")
    print("PARSE >sat:",sat_val_part_parsed,"-last:",last_val_part_parsed)
    print("F1_macro >sat:",sat_val_F1_macro,"-last:",last_val_F1_macro,">SATURED?",F1_macro_saturated," at epoch >",F1_macro_first_sat)
    print("F1_micro >sat:",sat_val_F1_micro,"-last:",last_val_F1_micro,">SATURED?",F1_micro_saturated," at epoch >",F1_micro_first_sat)
    print("================")

    F1_macro_broken_ratio=None
    if(F1_macro_saturated):
        print("F1_macro_first_sat>",F1_macro_first_sat)
        F1_macro_first_sat=int(F1_macro_first_sat)    
        if(sat_val_F1_macro>last_val_F1_macro):
            print("FORGETTING")
            F1_macro_forgetting=1
        if(len(F1_macro_broken_steps)>0 and (nb_epoch-F1_macro_first_sat)!=0):
            F1_macro_broken_ratio=len(F1_macro_broken_steps)/(nb_epoch-F1_macro_first_sat)

    F1_micro_broken_ratio=None
    if(F1_micro_saturated):
        print("F1_micro_first_sat>",F1_micro_first_sat)
        F1_micro_first_sat=int(F1_micro_first_sat)    
        if(sat_val_F1_micro>last_val_F1_micro):
            print("FORGETTING")
            F1_micro_forgetting=1
        if(len(F1_micro_broken_steps)>0 and (nb_epoch-F1_micro_first_sat)!=0):
            F1_micro_broken_ratio=len(F1_micro_broken_steps)/(nb_epoch-F1_micro_first_sat)
   

    parsed_broken_ratio=None
    if(parsed_saturated):  
        print("parsed_first_sat>",parsed_first_sat)
        parsed_first_sat=int(parsed_first_sat)
           
        if(parsed_first_sat>last_val_part_parsed):
            parse_forgetting=1
        if(len(parsed_broken_steps)>0 and (nb_epoch-parsed_first_sat)>0):
            parsed_broken_ratio=len(parsed_broken_steps)/(nb_epoch-parsed_first_sat)

    parse_sat=None
    if(parsed_saturated):
        parse_sat=1

    F1_macro_sat=None
    if(F1_macro_saturated):
        F1_macro_sat=1
    F1_micro_sat=None
    if(F1_micro_saturated):
        F1_micro_sat=1

    print("-> F1_macro_broken_ratio : ",F1_macro_broken_ratio)
    print("-> F1_micro_broken_ratio : ",F1_micro_broken_ratio)
    return {"F1_macro_sat":F1_macro_sat,"F1_macro_forgetting":F1_macro_forgetting,"F1_macro_first_sat":F1_macro_first_sat,"F1_macro_broken_ratio":F1_macro_broken_ratio,
    "F1_micro_sat":F1_micro_sat,"F1_micro_forgetting":F1_micro_forgetting,"F1_micro_first_sat":F1_micro_first_sat,"F1_micro_broken_ratio":F1_micro_broken_ratio,
    "parse_forgetting":parse_forgetting, "parsed_first_sat":parsed_first_sat,"parse_sat":parse_sat,"parsed_broken_ratio":parsed_broken_ratio}

#,"240ShadesOfSyntaxT5","CodeModels","OtherT5Models"



entity="celian-ringwald"
project="12ShadesOfRDF"
group="DS_turtleS_1inLine_1facto_T5_base_t5-base"
tag_name="T5_b_v_t_1_f"

best_score=0
path_best=None
print("================================")
#if(results_data[project][k]["tag_name"]=="T5_b_v_j"):
col_int=[
    "F1_micro_first_sat","F1_micro_broken_ratio",
    "F1_macro_first_sat","F1_macro_broken_ratio",
    "parsed_first_sat","parsed_broken_ratio"
    ]
col_means=["test_F1_micro",
"test_F1_macro",
"test_prec_macro",
"test_recall_macro",
"test_prec_micro",
"test_recall_micro",
"test_dist_edit",
"test_BLEU",
"test_part_parsed",
"test_part_subj_ok",
"test_part_valid",
"val_loss",
"carbon",
"nb_epoch",
"time",
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
for run in runs:
    name=run.name
    print(name)
    
    fold_tt="fold"+name.split("_")[-1]
    if fold_tt not in data_temp.keys():
        data_temp[fold_tt]={}
    data_temp[fold_tt]["nb_epoch"]=run.summary["epoch"]
    data_temp[fold_tt]["time"]=run.summary["_runtime"]/60
    
    for col in col_means:
        if(col not in data_temp[fold_tt].keys() or data_temp[fold_tt][col]==None):
            if(col in run.summary.keys()):
                data_temp[fold_tt][col]= run.summary[col]
            else:
                data_temp[fold_tt][col]=None
    ############ HISTO METRICS
    histo_metrics=getHistoryMetrics(run)
    #print(histo_metrics)
    for m in histo_metrics.keys():
        data_temp[fold_tt][m]=histo_metrics[m]
        
### MUST BE NOT NULL 

   
All_data=[]
All_data={"STRICT":[],"MEAN_WISE":[],"MEAN_NAIVE":[]}
data_aggr={"STRICT":{},"MEAN_WISE":{},"MEAN_NAIVE":{}}
data_temp2=[data_temp[k] for k in data_temp.keys()]

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
   if(nb==5): 
       data_aggr["STRICT"][col]=sum_/nb
   else:
       data_aggr["STRICT"][col]=None
   if(nb>0):
        data_aggr["MEAN_NAIVE"][col]=sum_/5
        data_aggr["MEAN_WISE"][col]=sum_/nb
   else:
        data_aggr["MEAN_NAIVE"][col]=None
        data_aggr["MEAN_WISE"][col]=None
        
for col in col_sums:
    sum_=0
    nb=0
    for d in data_temp2:
        if col in d.keys():
            if(d[col]!=None and str(d[col])!="NaN" and str(d[col])!="nan"):
                sum_+=d[col]
                nb+=1
            
    if(nb==5): 
        data_aggr["STRICT"][col]=sum_
    else:
        data_aggr["STRICT"][col]=None
    if(nb>0):
        data_aggr["MEAN_NAIVE"][col]=sum_
        data_aggr["MEAN_WISE"][col]=sum_
    else:
        data_aggr["MEAN_NAIVE"][col]=None
        data_aggr["MEAN_WISE"][col]=None
    
   # else:
    #    data_aggr[col]=None
print("====================================")
print(data_aggr)

print("====================================")
# and 'test_F1_macro' in data_aggr.keys()
for mode in data_aggr.keys():
    if(len(data_aggr[mode].keys())>0 ):
      test_F1_macro=data_aggr[mode]['test_F1_macro']
      test_part_parsed=data_aggr[mode]['test_part_parsed']
      test_part_subj_ok=data_aggr[mode]['test_part_subj_ok']/test_part_parsed
      data_aggr[mode]['test_part_subj_ok']=test_part_subj_ok
      test_part_valid=data_aggr[mode]['test_part_valid']/test_part_parsed
      data_aggr[mode]['test_part_valid']=test_part_valid
       
      score_all=test_F1_macro*test_part_parsed*test_part_subj_ok*test_part_valid
      data_extended=data_aggr[mode]
      for col in list(set(col_means+col_sums)):
          if(col not in data_extended.keys()):
              data_extended[col]=None
      data_extended["score"]=score_all
      data_extended["path_best"]=path_best
      data_extended["tag_name"]=tag_name
      data_extended["tokenizer_path"]=""
      data_extended["date"]=""
      data_extended["syntax"]="turtleS"
      data_extended["space"]=entity
      data_extended["add_vocab"]="True"
      data_extended["model_name_or_path"]=""
      All_data[mode].append(data_extended)
else:
    print("PB WITH >",tag_name)
df = pd.DataFrame.from_dict(All_data["STRICT"])
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
df3 = df2[["tag_name","R_TP", "R_CS", "R_SVT","F1-","F1+","P-","P+","R-","R+","B","lev","nb_epoch",
         "Sat_F1-","V_F1-","S_F1-","D_F1-","Sat_F1+","V_F1+","S_F1+","D_F1+",
         "Sat_RTP","V_RTP","S_RTP","D_RTP","Cc","Tt","Gg","val_loss",'tokenizer_path', 'date', 'syntax', 'space', 'add_vocab','path_best', 
       'model_name_or_path']]
df3.to_csv("//user/cringwal/home/Desktop/THESE/results_for_T5lost2.csv")  


