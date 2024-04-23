#!/usr/bin/env python
import argparse
import sys
import json 
import os 
import os.path
import wandb

################################ THIS SCRIPT IS USED TO EXTRACT FROM THE SERVER DATA RELATED TO MODELS AND MORE
# 
# python getCheckpointForTest.py  "/home/cringwald/EXP_env/output/" "/home/cringwald/EXP_env/results_exp"
#######################  INPUTS:
# - dir :   A DIRECTORY CONTAINING THE MODELS ARTIFACTS SAVED DURING & AFTER FINETUNING
# - output : path OF JSON FILE produced at the end

def parse_args():
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("dir")
    parser.add_argument("output")
    args=parser.parse_args()
    return args

def main():
    print("this is the main function")
    inputs=parse_args()
    mypath=inputs.dir

    print("=====================BEGIN============================")
    print("output path: ",mypath)
    print("output file: ",inputs.output)

    print("====================> GET WANDB RUN IDS")

    api = wandb.Api()
    ################" HERE  THE WANDB API KEY
    API_KEY="#######################"
    wandb.login(key=API_KEY)
    entities=["celian-ringwald","inria_test"]
    data_run={}
    for ent in entities:
        #data[ent]={}
        projects=api.projects(entity=ent)
        for p in projects:

             #### GET API DATA FOR LATTER MAPPING RUNS ID TO A SET OF FILES 
             runs=api.runs(
                path=ent+"/"+p.name
             )
             

             for r in runs:
                if(r.state=="finished"):
                    name=r.name
                    splitted=name.split('_')
                    fold=None
                    if(splitted[-1] in ["0","1","2","3","4"]):
                        fold=splitted[-1] 
                        data_run[r.id]={"space":ent,"project":p.name,"group":r.group,"name":name,"fold":fold}

                        for k in ["model_name_or_path","add_vocab","synthax","inline_mode","facto"]:
                            if(k in r.config.keys()):
                                data_run[r.id][k]=r.config[k]

    results_all={}
    ###############  GET FILES DATA
    dir_date_list=os.listdir(mypath)
    for dir_date in dir_date_list:
        path_date=os.path.join(mypath,dir_date)
        is_dir=os.path.isdir(path_date)
        if(is_dir):
            if(len(dir_date)==10 and dir_date[0:4]=="2024" and (int(dir_date[5:7])==1 and int(dir_date[8:10])>20) or int(dir_date[5:7])>1):
                print("-",dir_date)
                dir_hour_list=os.listdir(path_date)
                for dir_hour in dir_hour_list:
                    if(len(dir_hour)==8): 
                        CURRENT_PROJECT=None
                        tkz_dir_exist=False
                        data_summary=None
                        wandb_runs=[]
                        path_date_hour=os.path.join(path_date,dir_hour)
                        # print("--",path_date_hour)
                        path_date_hour_list=os.listdir(path_date_hour)
                        WANDB_data_path=os.path.join(path_date_hour,"wandb")
                        WANDB_data_exist=os.path.isdir(WANDB_data_path)
                        if(WANDB_data_exist):
                            dir_list_wandb=os.listdir(WANDB_data_path)
                            for wandb_dir in dir_list_wandb:
                                if("run" in wandb_dir):
                                    run_id=wandb_dir.split("-")[-1]
                                    if(run_id in data_run.keys()):
                                        wandb_runs.append(run_id)

                        # else:
                        #     print("not in WANDB")
                        MODEL_data_path=os.path.join(path_date_hour,"experiments")
                        MODEL_data_exist=os.path.isdir(MODEL_data_path)
                        data_summary_path=os.path.join(path_date_hour,"all_data.json")
                        data_summary_exist=os.path.isfile(data_summary_path)
                        CKPT_path=None
                        if(data_summary_exist==False):
                        #     print('DATA SUMMARY OK')
                        # else:
                            carbon_raw_path=os.path.join(path_date_hour,"emissions.csv")
                            carbon_raw_exist=os.path.isfile(carbon_raw_path)
                            # if(carbon_raw_exist):
                            #     print('RAW DATA SUMMARY OK')
                        if(MODEL_data_exist):
                            MODAL_data_list=os.listdir(MODEL_data_path)
                            for di in MODAL_data_list:

                                if(di=="experiments"):
                                    tokenizer_path=os.path.join(MODEL_data_path,"experiments")
                                    tkz_dir_exist=os.path.isdir(tokenizer_path)
                                    if(tkz_dir_exist):       

                                        dir_list7=os.listdir(tokenizer_path)
                                        if(len(dir_list7)==1):
                                            #print("TOKENIZER OK")
                                            exact_tokenizer_path1=os.path.join(tokenizer_path,dir_list7[0])
                                            #print("> path : ",exact_tokenizer_path1)
                                        # else:
                                        #     print("PB:TOKENIZER")
                                else:
                                    CURRENT_PROJECT=di
                                    CKPT_path=os.path.join(MODEL_data_path ,di)
                        if(CURRENT_PROJECT and CKPT_path and len(wandb_runs)>0): 
                            #print("CURRENT_PROJECT ? ",CURRENT_PROJECT)
                            ckpt_list=os.listdir(CKPT_path)
                            if(len(ckpt_list)>0):
                                #print("-",ckpt_list)
                                dict_checkpoint={}
                                for file_cpt in ckpt_list:
                                    ext_ok=file_cpt[len(file_cpt)-len(".ckpt"):len(file_cpt)]
                                    if(file_cpt[0]!="." and ext_ok==".ckpt"):
                                        clean_f_name=file_cpt.replace(".ckpt", "")
                                        if("epoch" in clean_f_name):
                                            data_cpt=clean_f_name.split("-epoch=")
                                            #print(data_cpt)
                                            if(len(data_cpt)==2):
                                                model_name_all=data_cpt[0].split("_")
                                                if(len(model_name_all[-1])==1):
                                                    fold=model_name_all[-1]
                                                    model_name=data_cpt[0].replace("_"+fold,"")
                                                    
                                                    #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",model_name)
                                                    data_cpt2=data_cpt[1].split("-val_loss=")
                                                    if(len(data_cpt2)==2):
                                                        epoch=data_cpt2[0]
                                                        val_loss=data_cpt2[1]
                                                        if(fold not in dict_checkpoint.keys()):
                                                            dict_checkpoint[fold]=[]
                                                            
                                                        dict_checkpoint[fold].append({"epoch": epoch,"val_loss":val_loss,"file_path":file_cpt})
                                        elif("last" in clean_f_name):
                                            if("-" in clean_f_name):
                                                split_v=clean_f_name.replace("v","").split("-")
                                                current_fold=split_v[1]
                                                
                                            else:
                                                current_fold="0"

                                            if(current_fold not in dict_checkpoint.keys()):
                                                dict_checkpoint[current_fold]=[]
                                            dict_checkpoint[current_fold].append({"epoch": "last","val_loss":"unk","file_path":file_cpt})

                                print("========================================")
                                print(">>>>",dir_hour)
                                #print("-wandb:",WANDB_data_exist)
                                have_check_point=len(dict_checkpoint.keys())>0 and len(dict_checkpoint.keys())==5
                                if(WANDB_data_exist):
                                    project=data_run[wandb_runs[0]]["project"]
                                    if(project not in results_all.keys()):
                                        results_all[project]={}

                                    group=data_run[wandb_runs[0]]["group"]
                                    if group and str(group)!="":
                                        if(group not in results_all[project].keys()):
                                            results_all[project][group]={"date":dir_date,"summary":None,"tokenizer_path":None,"ckpt":{},"runs":{}}

                                        for id_run in wandb_runs:

                                            fold_id=data_run[id_run]["fold"]
                                            name=data_run[id_run]["name"]
                                            space=data_run[id_run]["space"]
                                            results_all[project][group]["runs"][fold_id]={"id_run":id_run,
                                            "name":name,
                                            "space":space,
                                            "project":project,
                                            "group":group
                                            }


                                        ############ NAME BUILDING    
                                        tmp_model=data_run[id_run]["model_name_or_path"]
                                        temp_name=""
                                        results_all[project][group]["model_name_or_path"]=tmp_model
                                        if("bart" in tmp_model):
                                            temp_name="B"
                                            if("base" in tmp_model):
                                                temp_name+="_b"
                                            if("small" in tmp_model):
                                                temp_name+="_s"
                                            if("large" in tmp_model):
                                                temp_name+="_l"

                    

                                        elif("t5" in tmp_model):
                                            temp_name="T5"
                                            if("base" in tmp_model):
                                                temp_name+="_b"
                                            if("small" in tmp_model):
                                                temp_name+="_s"
                                            if("large" in tmp_model):
                                                temp_name+="_l"

                                        ################ VOCAB
                                        if("add_vocab" in data_run[id_run].keys()):
                                            add_vocab=data_run[id_run]["add_vocab"]
                                            if(add_vocab==True):
                                                 temp_name+="_v"
                                            results_all[project][group]["add_vocab"]=add_vocab
                                        syntax="None?"
                                        ################ SYNTHAX
                                        if("xml" in group):
                                            syntax="xml"
                                            temp_name+="_x"
                                        elif("ntriples" in group):
                                            syntax="ntriples"
                                            temp_name+="_n"
                                        elif("json-ld" in group):
                                            syntax="json-ld"
                                            temp_name+="_j"
                                        elif("turtle_" in group):
                                            syntax="turtle"
                                            temp_name+="_T"
                                        elif("list" in group):
                                            syntax="list"
                                            temp_name+="_l"
                                        elif("tags" in group):
                                            syntax="tags"
                                            temp_name+="_g"
                                        elif("turtleS" in group or "turtleLight" in group):
                                            syntax="turtleLight"
                                            temp_name+="_t"

                                        if("inline_mode" in data_run[id_run].keys()):
                                            inline=data_run[id_run]["inline_mode"]
                                            results_all[project][group]["inline_mode"]=inline
                                            if(inline==True and syntax in ["turtleLight"]):
                                                temp_name+="_1"

                                        if("facto" in data_run[id_run].keys()):                     
                                            facto= data_run[id_run]["facto"]              
                                            results_all[project][group]["facto"]=facto
                                            if(facto==True and syntax in ["turtleLight","list","tags"]):
                                                temp_name+="_f"
                                        results_all[project][group]["syntax"]=syntax
                                        results_all[project][group]["space"]=data_run[id_run]["space"]
                                        results_all[project][group]["tag_name"]=temp_name
                                        
                                        if(data_summary_exist):
                                            with open(data_summary_path,encoding="utf-8") as json_file:
                                               dataset_summary = json.load(json_file)
                                               results_all[project][group]["summary"]=dataset_summary

                                        if(tkz_dir_exist):
                                            results_all[project][group]["tokenizer_path"]=tokenizer_path
                                        if(have_check_point):
                                            results_all[project][group]["ckpt"]=dict_checkpoint


    print(results_all)    
    print("SAVE")
    with open(inputs.output, 'w', encoding='utf-8')  as f:
        json.dump(results_all, f)
    print("=====================END============================")


if __name__ == '__main__':
    main()
