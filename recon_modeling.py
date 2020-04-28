#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:55:40 2020

@author: callen
"""


#%%
from __future__ import print_function
import json
import cobra
from cobra.flux_analysis import (
    single_gene_deletion, single_reaction_deletion, double_gene_deletion,
    double_reaction_deletion)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import itertools
import os 

#%% model_info function
# note to self: recon3d model uses bigg ids for genes
# input: cobrapy model object
def model_info(model):

    print(str(len(model.reactions))+" Reactions")
    print(str(len(model.metabolites)) + "Metabolites")
    print(str(len(model.genes))+" Genes")
    print("Objective"+str(model.objective))
    modelinfo_dict = {}
    
    # Iterate through the the objects in the model
    modelinfo_dict["Reactions"]={}
    modelinfo_dict["Reactions"]["string"]="Reactions/n ---------/n"
    modelinfo_dict["Reactions"]["list"]=[]
    for x in model.reactions:
        modelinfo_dict["Reactions"]["string"]+=("%s : %s" % (x.id, x.reaction))
        modelinfo_dict["Reactions"]["list"].append([x.id, x.reaction])
    
    modelinfo_dict["Metabolites"]={}
    modelinfo_dict["Metabolites"]["string"]="Metabolites/n-----------/n"
    modelinfo_dict["Metabolites"]["list"]=[]
    for x in model.metabolites:
        modelinfo_dict["Metabolites"]["string"]+=('%9s : %s /n' % (x.id, x.formula))
        modelinfo_dict["Metabolites"]["list"].append([x.id, x.formula])
        
        
    modelinfo_dict["Genes"]={}
    modelinfo_dict["Genes"]["string"]="Genes/n-----/n"
    modelinfo_dict["Genes"]["list"]=[]
    for x in model.genes:
        associated_ids = (i.id for i in x.reactions)
        associated_rxns = ", ".join(associated_ids) 
        modelinfo_dict["Genes"]["list"].append({
            "gene_id" : x.id,
            "gene_name": x.name, 
            "reactions": associated_rxns})
        modelinfo_dict["Genes"]["string"]+=("%s is associated with reactions: %s /n" %
              (x.id, "{" + associated_rxns + "}"))
    return modelinfo_dict


#%% load_what_model
# loads cobra model of a file based on its filetype
# inputs: model_file -- string or path variable to file of interest
#         filetype -- string, either 'json' or 'mat'
rootdir = ""
def load_what_model(model_file, filetype=None):
    model_filetype = None
    # yanking out the file extension.
    # accounts for absolute/relative path specification
    filestring_split = (model_file.split("/"))[-1].split(".")
    function_switcher = {
            "json" : lambda x: cobra.io.json.load_json_model(x),
            "mat" : lambda x: cobra.io.load_matlab_model(x)
        }
    # QC on parameter specifications by user. 
    # user can either throw in a ".[json/mat]" file or 
    # use an extension-less file, but specify what file type it is
    if len(filestring_split) >= 2:
        # some people put '.'s in their dir names
        # so looking at last one generically just in case
        model_filetype=filestring_split[-1]
    elif filetype is not None:
        # sometimes people have transcended beyond the 
        # concept of file extensions. We must please all,
        # including those that we cannot see since they are
        # in a different astral plane.
        model_filetype=filetype.lower()
    else:
        # if the user doesn't wield the function correctly 
        # based on above
        exit("""Error: need to either have a proper
             extension[.mat or .json] or set filetype
             parameter as a string, either 'json' or 'mat'.""")
    reading_function = function_switcher[model_filetype]
    return reading_function(model_file)

#%% reaction modulator

def rxn_modulate(model, 
                 rxn_2mod, 
                 start, 
                 end, 
                 rxns_monitored,
                 which_bound="upper", 
                 num_steps=100, 
                 rxn_list=None):
    model_b = model
    bound_list = np.linspace(start, end, num_steps)
    my_sols_dict = {r : [] for r in rxns_monitored}
    for f in range(num_steps):
        if which_bound == "upper":
            model_b.reactions.get_by_id(rxn_2mod).upper_bound = bound_list[f]
        else:
            model_b.reactions.get_by_id(rxn_2mod).lower_bound = bound_list[f]
        this_sol = model_b.optimize()
        for r in rxns_monitored:
            my_sols_dict[r].append(this_sol.fluxes[r])
    return(my_sols_dict)

#%% set wd based on user's file system/working copy
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%% load model

# load recon model
print("loading model....")
this_model = load_what_model("Recon3D.json")
# get model info object
this_model_info = model_info(this_model)

#%% unperturbed solution

this_solution = this_model.optimize()
unperturbed_growth = this_solution.objective_value


#%% get drug targets

# getting all the gene names
genie_dict = {x["gene_name"]:x["gene_id"] for x in this_model_info["Genes"]["list"]}
with open("genie_dict.json","w") as gd:
    json.dump(genie_dict, gd)
# a list of the drug targets from the covid ppi paper
# [help me]
drug_targets = [
    "EIF4E2", "FKBP15", "NUP210", "HDAC2", "ATP6AP1", "SIGMAR1",
    "NEK9", "EIF4H", "NUP62", "NUP214", "NUP88", "NUP58", "NUP54",
    "PTGES2", "COMT", "NDUFAF2", "GLA", "IMPDH2", "MRPS5", "MRPS25",
    "MRPS27", "MRPS2", "RIPK1", "CEP250", "PRKACA", "TBK1", "BRD2", "BRD4",
    "LARP1", "CSNK2B", "CSNK2A2", "NDUFAF1", "NDUFB9", "ABCC1", "F2RL1", "TMEM97",
    "EDEM3", "ERLEC1", "OS9", "UGGT2", "ERO1B", "SIL1", "HYOU1", "NGLY1", "TOR1A",
    "FOXRED2", "SDF2", "LOX", "PLOD2", "FKBP10", "FKBP7", "DNMT1", "NUP98", "RAE1",
    "MARK2", "MARK3", "DCTPP1", "ELOC", "ZYG11B", "REX1", "CUL2"    
]
# finding the genes that are drug targets and are present in 
# recon model
drug_targets_model = list(set(drug_targets).intersection(set(genie_dict.keys())))
drug_targets_model_ids = [genie_dict[i] for i in drug_targets_model]


#%% parent and children nodes

# fetching all the sars-cov2 gene-gene interactions from the given csv file
interactions = pd.read_csv(
        rootdir+"project_support/Network_Gene_Gene_Reaction.csv"
    )
# so this file is made kinda funny. for any sars-cov2 interaction, sars-cov2 
# is the source, and the gene(s) it targets get listed in the 'TYPE' column
# so we simply need to pull the gene-gene interactions from this file and 
# then isolate the TYPE values. sometimes the target value is just a single 
# letter, which I take to be part of that set of genes they've named.

# getting all gene-gene interactions as pandas frame
interactions_gene_gene = interactions.dropna(subset=["IS_GENE_GENE"])
# get unique names for the direct targets of sars-cov2 [parents] and those
# the parents affect[children] for querying online sources
sars_cov2_targets_parents = list(set(interactions_gene_gene["TARGET"]))
sars_cov2_targets_children = list(set(interactions_gene_gene["TYPE"]))

# debug block. set to True if you want to see the targets
if False:
    print("PARENTS")
    print(sars_cov2_targets_parents)
    print("CHILDREN")
    print(sars_cov2_targets_children)

#%% Finding the reactions the drug targets are associated with

# some genes are associated with more than 1 rxn
# and the reactions property will be comma separated if so
# so just assume they all are associated with >1 rxn
# then "unlist" after
drug_target_rxn_dictlist = [x["reactions"].split(",") for x in this_model_info["Genes"]["list"] if x["gene_name"] in drug_targets]
flat_drug_target_rxn = []
for x in drug_target_rxn_dictlist:
    for r in x:
        flat_drug_target_rxn.append(r.strip())
#get unique rxn ids
flat_drug_target_rxn = list(set(flat_drug_target_rxn))
#%% single-gene KO
# single gene KOs
print("single gene kos")
sars_geneko_single = single_gene_deletion(this_model, 
                                          gene_list = (drug_targets_model_ids))


#%% double gene KOs
if False:
    print("double gene kos")
    sars_geneko_double = double_gene_deletion(this_model, 
                                              gene_list = drug_targets_model_ids)

#%% oxygen modulation
o2_rids = [r.id for r in this_model.metabolites.o2_m.reactions]
o2_names = [this_model.reactions.get_by_id(r).name for r in o2_rids]

# This reaction is responsible for oxygen diffusion
o2_diffusion_id = "O2tm"
o2_diffusion = this_model.reactions.get_by_id(o2_diffusion_id)
o2_diffusion_upper_default = o2_diffusion.upper_bound
o2_diffusion_lower_default = o2_diffusion.lower_bound
print("Modulating bound of " + o2_diffusion_id)
o2_diffusion_mods = rxn_modulate(this_model, 
                                 o2_diffusion_id,
                                 o2_diffusion_upper_default, 
                                 0,
                                 flat_drug_target_rxn,
                                 "upper", 
                                 1000)