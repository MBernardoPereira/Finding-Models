#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bernardo
"""
import project_code as pc

T1=[[1],[4],[5],[8]]
T2=[[2],[3],[6],[7]]
T3=[[1],[4],[5],[7]]
T4=[[1],[3],[5],[8]]
T5=[[2],[3],[5],[7]]
T6=[[1],[3],[6],[7]]
T7=[[1],[3],[5],[7]]

all_training_examples = [T1, T2, T3, T4, T5, T6, T7]

def full_crossing(a,b):
    c=[]
    for ai in a:
        for bi in b:
            ci=set(ai).union(set(bi))
            if ci not in c:
                c.append(ci)
    return c

#is atom a wider than atom b?
#assuming a is different from b:
def is_wider(a,b):
    for constant in b:
        if constant not in a:
            return False
    return True

def remove_redundant_atoms(model):
    new_model=model.copy()
    
    for atom in model:
        arr = []
        for constant in atom:
            trig = 1
            #atoms that are in that same constant
            aux_atoms = [phi for phi in model if set([constant]).issubset(set(phi))]
            aux_atoms.remove(atom)
            for aux_atom in aux_atoms:
                if is_wider(atom, aux_atom):
                    trig = 0
                    arr.append(trig)
            
            if trig!=0:       
                arr.append(1)

        if sum(arr) == 0:
            new_model.remove(atom)
        
        model=new_model
    return new_model

def create_freest_model(training_examples):
    freest_model = [set([1]),set([2]),set([3]),set([4]),set([5]),
                    set([6]),set([7]),set([8]),set(['p'])]
    
    #doing the same procedure for every training image
    for example in training_examples:
        
        #Checking which atoms from the current model are in the given image
        atoms_in_example = []
        for atom in freest_model:
            for constant in example:
                if set(constant).issubset(set(atom)):
                    if atom not in atoms_in_example:
                        atoms_in_example.append(atom)
        
        #print('freest_model ',freest_model)
        #print('atoms_in_example ',atoms_in_example)
        
        #Finding the atoms from the current model that are in p and not in the image
        atoms_in_p_and_not_in_example=[]
        for atom in freest_model:
            if (atom not in atoms_in_example) and set(['p']).issubset(set(atom)):
                atoms_in_p_and_not_in_example.append(atom)
        
        print("                ")
        print('Atoms_in_Example')
        print(atoms_in_example)
        
        print('Atoms_in_p_and_not_in_Example')
        print(atoms_in_p_and_not_in_example)
        
        
        #If we have atoms to do the full-crossing we do it
        if atoms_in_p_and_not_in_example != []: 
            new_atoms = full_crossing(atoms_in_example, atoms_in_p_and_not_in_example)
                
            print('NEW ATOMS')
            print(new_atoms)
            print('nr_new_atoms: ', len(new_atoms))
        
            #remove the atoms with 'p' that were used in the full crossing
            for atom in atoms_in_p_and_not_in_example:
                freest_model.remove(atom)
           
            #add the new atoms in 'p' to the new model
            for atom in new_atoms:
                freest_model.append(atom)
            
            print("MODEL BEFORE REMOVING REDUNDANT")
            print(freest_model)
            freest_model = remove_redundant_atoms(freest_model)
            
            aux=[]
            for el in freest_model:
                if el in new_atoms:
                    aux.append(el)
        else:
            print("atoms_in_p_and_not_in_example is empty")
        
        
        print("NEW ATOMS AFTER REMOVING REDUNDANT ATOMS")
        print(aux)

        print("MODEL AFTER REMOVING REDUNDANT ATOMS")
        print(freest_model)
        
    #order the atoms in the model according to their size
    freest_model = sorted(freest_model, key=len)      
    return freest_model

             
def mhs_hitman(model):
    from pysat.examples.hitman import Hitman
    
    h = Hitman(solver='m22', htype='lbx')
        
    for el in model:
        h.hit(el)
            
    #showing and blocking the hitting sets found
    set_of_hitting_sets=[]
    cur_set = h.get()
    while cur_set != None:
        set_of_hitting_sets.append(cur_set)
        h.block(cur_set)
        cur_set=h.get()
        
    set_of_hitting_sets.sort(key=len)

    return set_of_hitting_sets


#Checking that all the permutations of the training examples produce the same model
"""
from itertools import permutations
l = list(permutations(all_training_examples))

all_dif_models=[]
for perm in l:
    new_model = create_freest_model(perm)
    if new_model not in all_dif_models:
        all_dif_models.append(new_model)


all_dif_models_without_initial_atoms = []
for m in all_dif_models:
    res = []
    for el in m:
        if len(el)>1:
            res.append(el)
    all_dif_models_without_initial_atoms.append(res)
"""


#Plotting the hitting sets from the model outputed
"""
test_model = [{1, 2}, {1, 3}, {3, 4}, {3, 5}, {5, 6}, {1, 7}, {5, 7}, {7, 8}]

test_submodel_ = [ {1, 3}, {3, 5}, {1, 7}, {5, 7}]

l=mhs_hitman(test_model)

def translate(list_to_translate):
    for i in range(len(list_to_translate)):
        if list_to_translate[i] == 1 : list_to_translate[i] = 0
        if list_to_translate[i] == 2 : list_to_translate[i] = 4
        if list_to_translate[i] == 3 : list_to_translate[i] = 1
        if list_to_translate[i] == 4 : list_to_translate[i] = 5
        if list_to_translate[i] == 5 : list_to_translate[i] = 2
        if list_to_translate[i] == 6 : list_to_translate[i] = 6
        if list_to_translate[i] == 7 : list_to_translate[i] = 3
        if list_to_translate[i] == 8 : list_to_translate[i] = 7
    return list_to_translate

print(l)
for el in l:
    el=translate(el)
    at=pc.converting_explicit_list_to_binary_list(el,2)
    pc.plot_atom(at)
    
"""


#%%

#Creation of the Freest Model by giving all the training examples

#ATTENTION: This execution will print several pieces of information along the training process 
#so that it is possible to check what is happening with the model along the way
model = create_freest_model(all_training_examples)




