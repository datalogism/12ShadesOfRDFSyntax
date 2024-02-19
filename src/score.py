#!/usr/bin/env python3

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import numpy as np 
from sklearn import metrics
from torchmetrics.text import SacreBLEUScore
import score_fct as scr_fct
from fast_edit_distance import edit_distance
import lmppl

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args





def re_score_withShape(pred_relations, gt_relations, format_, syntax_conf, shape):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>RESCORE")
    print(shape)
    get_prop = """
    SELECT DISTINCT ?target_prop ?datatype
    WHERE {
        ?a sh:path ?target_prop.
        ?a sh:datatype ?datatype
    }"""
    qres = shape.query(get_prop)


    relations_map={}    
    for row in qres:
       if("#" in row[0]):
        relations_map[str(row[0]).split("#")[-1]]={"ns":row[0],"dt":row[1]}
       else:
        relations_map[str(row[0]).split("/")[-1]]={"ns":row[0],"dt":row[1]}

    relations_all=list(relations_map.keys()) 

    scores = {rel: {"tp": 0, "fp": 0, "fn": 0,"bleu":None} for rel in relations_all + ["ALL"]}

    bleu_scores = {rel: [] for rel in relations_all }


    n_sents = len(gt_relations)
    nb_rel = []
    nb_found = []
    n_subj_ok = 0
    n_datatype_ok = 0 ################ ONLY FOR DATATYPED FORMAT 
    n_parsed = 0
    nb_valshape = 0
    n_typeEnt_ok = 0
    edit_dist = [] 
    pred_relations_clean=scr_fct.cleanDecoded(pred_relations)
    gt_relations_clean=scr_fct.cleanDecoded(gt_relations)

    for p_sent, g_sent in zip(pred_relations_clean, gt_relations_clean):
        p_sent = str(p_sent)
        g_sent = str(g_sent)

        #parsed_pred=False
        type_pred = []
        type_gold = []
        rel_found = []
        rel_gold = []
        dict_rel_gt = {}
        dict_rel_pred = {}
        subj_gt = ""
        subj_pred = ""
        # print(">>>>>><list_rel_gt")
        list_rel_gt, parsed_gt = scr_fct.toListRel(g_sent,format_,syntax_conf["facto"])
        list_rel_pred, parsed_pred = scr_fct.toListRel(p_sent,format_,syntax_conf["facto"])

       
        # print(">>>>>><list_rel_pred")
        # print(list_rel_pred)
        # print("FROM")
        # print(p_sent)
       
        if(parsed_pred == False and parsed_gt == True):
            edit_dist.append(edit_distance(p_sent,g_sent))

        if(parsed_gt and isinstance(list_rel_gt, list) and len(list_rel_gt)>0):   
            #print("--list_rel_gt ok")
            subj_gt=str(list_rel_gt[0][0])
            subj_gt=subj_gt.replace("http://dbpedia.org/resource/","")

            type_current = None

            for triple in list_rel_gt:
                if(len(triple) == 3):
                    rel = str(triple[1])
                    val = str(triple[2])
                    rel_2=[sub for sub in relations_all if sub in rel] ### CHECK IT ?
                    if(len(rel_2) == 1 and rel_2[0]!=""):
                        rel_new=rel_2[0]
                        if(rel_new not in dict_rel_gt.keys()):
                            dict_rel_gt[rel_new] = []
                        if(val not in dict_rel_gt[rel_new]):
                            dict_rel_gt[rel_new].append(val)
                #else:
                #    print("")
            
            
            nb_rel.append(len(list_rel_gt))

            if(parsed_pred and isinstance(list_rel_pred, list) and len(list_rel_pred)>0):

                ## CHECK TYPE
                pred_type = []
                gold_type = [str(rel[2]) for rel in list_rel_gt if str(rel[1]) in ["rdfs:type","type","a",'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']]
                try:
                    pred_type = [str(rel[2]) for rel in list_rel_pred if str(rel[1]) in ["rdfs:type","type","a",'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']]
                except:
                    print("PB WITH >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print(list_rel_pred)

                if(len(gold_type) > 0 and len(pred_type) > 0 and gold_type[0] == pred_type[0]):
                    n_typeEnt_ok += 1
                    
                valid=False
                try:
                    rdf_ = scr_fct.list_ToRDF2(list_rel_pred,relations_map,shape)

                    valid = scr_fct.validateTriple(rdf_,shape)
                except:
                    print("VALIDATION PB")
                #print(">>>>>>>>>>> VALID ? "+str(valid))
                if(valid == True):
                    nb_valshape += 1

                subj_pred = str(list_rel_pred[0][0])
                subj_pred = subj_pred.replace("http://dbpedia.org/resource/","")
                
                for triple in list_rel_pred:
                    if(len(triple) == 3):
                        rel = str(triple[1])
                        val = str(triple[2])
                        rel_2 = [sub for sub in relations_all if sub in rel]
                        if(len(rel_2) == 1 and rel_2[0]!=""):
                            rel_new = rel_2[0]
                            if(rel_new not in dict_rel_pred.keys()):
                                dict_rel_pred[rel_new] = []
                            if(val not in dict_rel_pred[rel_new]):
                                dict_rel_pred[rel_new].append(val)
                    else:
                        print(">>>>>>>>>>> PB with triple",triple)

                for rela in relations_all:
                    if(rela in dict_rel_gt.keys()):
                        rel_gold.append(rel)
                        if(rela in dict_rel_pred.keys()):
                            for val in dict_rel_pred[rela]:
                                ########## CHECK DATATYPES FOR BLEU
                                if("string" in relations_map[rela]["dt"] ):
                                    sacre_bleu = SacreBLEUScore(tokenize='char')
                                    bscore = sacre_bleu([val], [dict_rel_gt[rela]])
                                    bleu_scores[rela].append(float(bscore))

                                if(val in dict_rel_gt[rela]):    
                                    scores[rela]["tp"] += 1
                                    rel_found.append(rel)
                                else:
                                    scores[rela]["fp"] += 1
                        else:
                            scores[rela]["fn"] += 1

                nb_found.append(len(rel_found))
                
                if(subj_gt != "" and subj_gt == subj_pred ):
                    n_subj_ok += 1
                if(parsed_pred):
                    n_parsed += 1 
            else:
                print(">>>>>>>>>>> PB with predictions")
                print(p_sent)
                print(">>>>>>>>>>> must look like ")
                print(list_rel_gt)    
                print(">>>>>>>>>>> list ")
                print(list_rel_pred)      
                print("GT >>>>>>>>>>")
                print(g_sent)
                 
            
        else:
            print(">>>>>>>>>>> PB with ground truth triples")
            print(g_sent)
            print(">>>>>>>>>>> list ")
            print(list_rel_gt)
        #    parsed_pred=False

    ######### BLEU AVG
    for rela in bleu_scores.keys():
        if(len(bleu_scores[rela])>0):
            scores[rela]["bleu"] = np.round(np.mean(bleu_scores[rela]), 4) 

    for rel_type in scores.keys():
       if scores[rel_type]["tp"]:
           scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
           scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
       else:
           scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

       if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
           scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                   scores[rel_type]["p"] + scores[rel_type]["r"])
       else:
           scores[rel_type]["f1"] = 0
    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relations_all])
    fp = sum([scores[rel_type]["fp"] for rel_type in relations_all])
    fn = sum([scores[rel_type]["fn"] for rel_type in relations_all])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0


    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relations_all])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relations_all])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relations_all])
    scores["ALL"]["BLEU"] = np.mean([scores[rela]["bleu"] for rela in scores.keys() if scores[rela]["bleu"] is not None])
    scores["ALL"]["AVG_EditDist"]=np.mean(edit_dist)

    #print(f"RE Evaluation in *** {mode.upper()} *** mode")
    n_rels = sum(nb_rel)
    n_found = sum(nb_found)
    part_parsed = n_parsed / n_sents
    part_subj_ok = n_subj_ok / n_sents
    part_type_ok = n_typeEnt_ok / n_sents
    part_valid = nb_valshape / n_sents
    print(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,
                                                                                             tp))
    print(
        "\tALL\t TP: {};\tFP: {};\tFN: {};\tBLEU: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"],
            scores["ALL"]["BLEU"]))
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro) \tAVG_EditDist: {:.2f} (norm)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"],
            scores["ALL"]["AVG_EditDist"]))

    for rel_type in relations_all:
        if(scores[rel_type]["bleu"] != None):
 
            print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\tbleu: {:.2f};\t{}".format(
                rel_type,
                scores[rel_type]["tp"],
                scores[rel_type]["fp"],
                scores[rel_type]["fn"],
                scores[rel_type]["p"],
                scores[rel_type]["r"],
                scores[rel_type]["f1"],
                scores[rel_type]["bleu"],
                scores[rel_type]["tp"] +
                scores[rel_type]["fp"]))
        else:
            print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
                rel_type,
                scores[rel_type]["tp"],
                scores[rel_type]["fp"],
                scores[rel_type]["fn"],
                scores[rel_type]["p"],
                scores[rel_type]["r"],
                scores[rel_type]["f1"],
                scores[rel_type]["tp"] +
                scores[rel_type]["fp"]))

    print(">>>>>>>>>>>>>>>>>>>>>>> PARSING OK >",part_parsed," %")
    print(">>>>>>>>>>>>>>>>>>>>>>> SUBJ OK >",part_subj_ok," %")
    print(">>>>>>>>>>>>>>>>>>>>>>> SHACL OK >",part_valid," %")
    print(">>>>>>>>>>>>>>>>>>>>>>> PART TYPE OK >>>",part_type_ok,"%")
    return scores, part_parsed, part_subj_ok, part_valid

if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: {} in gold vs {} in prediction".format(len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

