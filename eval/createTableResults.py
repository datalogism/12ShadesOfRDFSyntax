import json 
import pandas as pd

with open("//user/cringwal/home/Desktop/THESE/results_april2024.json",encoding="utf-8") as json_file:
   results_data = json.load(json_file) 

import wandb
import time
api = wandb.Api()
API_KEY="################"
wandb.login(key=API_KEY)


####################" THIS FUNCTION ALLOWS TO OBTAIN METRICS BASED ON EVALUATION DURING TRAINING
def getHistoryMetrics(run):

    ###################### INIT EVERY VALUES
    #### F1 MACRO
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

    ################################## ITERATE THE HISTORY
    for epoch in complete_hist.keys():
        
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
                    if(val_part_parsed>0.9):
                        parsed_saturated=True
                        parsed_first_sat=int(epoch)
                        sat_val_part_parsed=val_part_parsed
                ############ ALREADY SATURATE > BROKEN STEPS ? 
                elif(parsed_saturated and val_part_parsed < sat_val_part_parsed ):
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
                    if(val_F1_macro>90):
                        F1_macro_saturated=True
                        F1_macro_first_sat=int(epoch)
                        sat_val_F1_macro=val_F1_macro
                ############ ALREADY SATURATE > BROKEN STEPS ? 
                elif(F1_macro_saturated and val_F1_macro < sat_val_F1_macro):
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
                    if(val_F1_micro>90):
                        F1_micro_saturated=True
                        F1_micro_first_sat=int(epoch)
                        sat_val_F1_micro=val_F1_micro
                ############ ALREADY SATURATE > BROKEN STEPS ? 
                elif(F1_micro_saturated and val_F1_micro < sat_val_F1_micro):
                    F1_micro_broken_steps.append(int(epoch))
                    
    ################## COMPUTE STABILITY
    ##### F1 MACRO
    F1_macro_broken_ratio=None
    if(F1_macro_saturated):
        F1_macro_first_sat=int(F1_macro_first_sat)    
        if(sat_val_F1_macro>last_val_F1_macro):
            F1_macro_forgetting=1
        if(len(F1_macro_broken_steps)>0 ):
            F1_macro_broken_ratio=len(F1_macro_broken_steps)/(nb_epoch)
    ##### F1 MICRO
    F1_micro_broken_ratio=None
    if(F1_micro_saturated):
        F1_micro_first_sat=int(F1_micro_first_sat)    
        if(sat_val_F1_micro>last_val_F1_micro):
            F1_micro_forgetting=1
        if(len(F1_micro_broken_steps)>0 ):
            F1_micro_broken_ratio=len(F1_micro_broken_steps)/(nb_epoch)
   
    
    ##### PARSING
    parsed_broken_ratio=None
    if(parsed_saturated):  
       # print("parsed_first_sat>",parsed_first_sat)
        parsed_first_sat=int(parsed_first_sat)
           
        if(parsed_first_sat>last_val_part_parsed):
            parse_forgetting=1
        if(len(parsed_broken_steps)>0 ):
            parsed_broken_ratio=len(parsed_broken_steps)/(nb_epoch)

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
        
    return {"F1_macro_sat":F1_macro_sat,"F1_macro_forgetting":F1_macro_forgetting,"F1_macro_first_sat":F1_macro_first_sat,"F1_macro_broken_ratio":F1_macro_broken_ratio,
    "F1_micro_sat":F1_micro_sat,"F1_micro_forgetting":F1_micro_forgetting,"F1_micro_first_sat":F1_micro_first_sat,"F1_micro_broken_ratio":F1_micro_broken_ratio,
    "parse_forgetting":parse_forgetting, "parsed_first_sat":parsed_first_sat,"parse_sat":parse_sat,"parsed_broken_ratio":parsed_broken_ratio}


selected_project=["12ShadesOfRDF"]
All_data={"STRICT":[],"MEAN_WISE":[],"MEAN_NAIVE":[]}

for project in selected_project:
    ################### FOR EACH MODELS
    for k in results_data[project].keys(): 
        ############# INITIATE VAR
        best_score=0
        path_best=None   
        tag_name=results_data[project][k]["tag_name"]
        data_temp={}
        best_fold=None
        
        #### MAKE DIFFERENCE BETWEEN AGGREGATION MODE
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
        col_sums=["parse_forgetting","parse_sat","F1_micro_forgetting","F1_micro_sat","F1_macro_forgetting","F1_macro_sat"]
        
        
        ###################### IF WE RETRIEVE DATA FROM SERVER
        if(results_data[project][k]["summary"] and len( results_data[project][k]["summary"].keys())>0):
            ############# FOR EACH FOLDS
            for fold in results_data[project][k]["summary"].keys():
                ########## CARBON DATA ARE SAVED ON SERVER > WE CATCH HERE ONLY TRAINING TIME COST
                carbon=results_data[project][k]["summary"][fold]["carbon_data"]["train_emissions"]
                t_d=results_data[project][k]["summary"][fold]["test_data_last_step"]
                temp=t_d[0]
                temp["carbon"]=carbon
                data_temp[fold]=temp
              
                test_F1_macro=temp['test_F1_macro']
                test_part_parsed=temp['test_part_parsed']
                test_part_subj_ok=temp['test_part_subj_ok']
                test_part_valid=temp['test_part_valid']
                score=test_F1_macro*test_part_parsed*test_part_subj_ok*test_part_valid
                ######### WE SAVE HERE THE BETTER FOLD IN TERM OF Gg 
                if(score>best_score):
                    best_score=score
                    path_best=results_data[project][k]["summary"][fold]["best_model_path"]
            ######## IF WE DO NOT HAVE A BEST FOLD WE JUST GET THE LAST CHKPOINT
            if(path_best==None):
                for fold in results_data[project][k]["ckpt"].keys() :
                    if(path_best==None):
                        for ckpt in  results_data[project][k]["ckpt"][fold]:
                            if(ckpt["epoch"]=="last" and path_best==None):
                                print("FOUND ALT PATH")
                                path_best=ckpt["file_path"]
                            
        
        ################### FOR EACH MODELS
        for id_ in results_data[project][k]["runs"].keys():
            
            entity=results_data[project][k]["runs"][id_]["space"]
            project=results_data[project][k]["runs"][id_]["project"]
            run_id=results_data[project][k]["runs"][id_]["id_run"]
            
            ############### GET DATA FROM WANDB
            run = api.run(entity+"/"+project+"/"+run_id)
            
            fold_tt="fold"+id_
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
            
            ##########################################"" FIX HERE THE FACT DIST ISN'T COMPUTED WHEN AVG=0
            for f in data_temp.keys():
                    if(str(data_temp[f]["test_dist_edit"]).lower()=="nan"):
                        
                        data_temp[f]["test_dist_edit"]= 0
                        
        ################### WE MAKE HERE THE DIFFENCE BETWEEN AGGREGATE OPTIONS 
        data_aggr={"STRICT":{},"MEAN_WISE":{},"MEAN_NAIVE":{}}
        data_temp2=[data_temp[k] for k in data_temp.keys()]

        ########## COMPUTE AVERAGE         
        for col in col_means:
           sum_=0
           nb=0
           ok=True
           for d in data_temp2:
               if col in d.keys():
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
                
               
        
        ########## COMPUTE SUM
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
            
        ############ FOR EACH MODE WE FINALISE THE BUILD OF THE TABLE  
        for mode in data_aggr.keys():
            if(len(data_aggr[mode].keys())>0 ):
            
              test_F1_macro=data_aggr[mode]['test_F1_macro']
              test_part_parsed=data_aggr[mode]['test_part_parsed']
              
              ########### RECOMPUTATION of R_SVT & R_CS depending of R_TP
              test_part_subj_ok=data_aggr[mode]['test_part_subj_ok']/test_part_parsed
              data_aggr[mode]['test_part_subj_ok']=test_part_subj_ok
              test_part_valid=data_aggr[mode]['test_part_valid']/test_part_parsed
              data_aggr[mode]['test_part_valid']=test_part_valid
              ######### COMPUTE Gg Score
              score_all=test_F1_macro*test_part_parsed*test_part_subj_ok*test_part_valid
              ######## BIND TOGETHER ALL THE DATA
              data_extended=data_aggr[mode]
              for col in list(set(col_means+col_sums)):
                  if(col not in data_extended.keys()):
                      data_extended[col]=None
              data_extended["score"]=score_all
              ######## AND ADD DATA RELATED TO files / naming conventions
              data_extended["path_best"]=path_best
              data_extended["tag_name"]=tag_name
              data_extended["tokenizer_path"]=results_data[project][k]["tokenizer_path"]
              data_extended["date"]=results_data[project][k]["date"]
              data_extended["syntax"]=results_data[project][k]["syntax"]
              data_extended["space"]=results_data[project][k]["space"]
              data_extended["add_vocab"]=results_data[project][k]["add_vocab"]
              data_extended["model_name_or_path"]=results_data[project][k]["model_name_or_path"]
              All_data[mode].append(data_extended)
       
######### SAVE A FILE BY AGGR MODE
for k in All_data.keys():
    df = pd.DataFrame.from_dict(All_data[k])
    ###### RENAME COLUMNS
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
    ###### SORT COLUMNS
    df3 = df2[["tag_name","R_TP", "R_CS", "R_SVT","F1-","F1+","P-","P+","R-","R+","B","lev","nb_epoch",
             "Sat_F1-","V_F1-","S_F1-","D_F1-","Sat_F1+","V_F1+","S_F1+","D_F1+",
             "Sat_RTP","V_RTP","S_RTP","D_RTP","Cc","Tt","Gg","val_loss",'tokenizer_path', 'date', 'syntax', 'space', 'add_vocab','path_best', 
           'model_name_or_path']]
    ###### SORT ROWS
    df4=df3.sort_values('Gg', ascending=False)
    df4.to_csv("//user/cringwal/home/Desktop/THESE/results_april2024_"+k+".csv")  

