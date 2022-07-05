#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bernardo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pysat.examples.hitman import Hitman


#%%
#Here the sets of atoms are created (or imported)

model1=np.array([1,0,0, 1,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model2=np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model3=np.array([1,0,0, 1,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model4=np.array([1,0,0, 0,1,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model5=np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model6=np.array([1,0,0, 0,1,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model7=np.array([1,0,0, 0,0,1, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model8=np.array([1,0,0, 0,0,1, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model9=np.array([1,0,0, 0,0,1, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model10=np.array([0,1,0, 1,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model11=np.array([0,1,0, 1,0,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model12=np.array([0,1,0, 1,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model13=np.array([0,1,0, 0,1,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model14=np.array([0,1,0, 0,1,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model15=np.array([0,1,0, 0,1,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model16=np.array([0,1,0, 0,0,1, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model17=np.array([0,1,0, 0,0,1, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model18=np.array([0,1,0, 0,0,1, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model19=np.array([0,0,1, 1,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model20=np.array([0,0,1, 1,0,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model21=np.array([0,0,1, 1,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model22=np.array([0,0,1, 0,1,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model23=np.array([0,0,1, 0,1,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model24=np.array([0,0,1, 0,1,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

model25=np.array([0,0,1, 0,0,1, 1,0,0, 0,0,0, 0,0,0, 0,0,0])
model26=np.array([0,0,1, 0,0,1, 0,1,0, 0,0,0, 0,0,0, 0,0,0])
model27=np.array([0,0,1, 0,0,1, 0,0,1, 0,0,0, 0,0,0, 0,0,0])

#all the atoms from the exact 3by3 model
data_exact_atoms_3by3 = np.vstack((model1,model2,model3,model4,model5,model6,model7,
                                   model8,model9,model10,model11,model12,model13,model14,
                                   model15,model16,model17,model18,model19,model20,model21,
                                   model22,model23,model24,model25,model26,model27))

#37 atoms from the 4by4 model (a subset of the model)
data_37_atoms_subset_of_the_model = np.load('/Users/Bernardo/Desktop/MMAC/2sem/AMLProject/subset_of_model.npy')
data_37_atoms_subset_of_the_model = data_37_atoms_subset_of_the_model[:,1:]

#all the atoms from the exact 4by4 model
data_exact_atoms_4by4=[]
for i in range(0,4):
   	for j in range(4,8):
   		for k in range(8,12):
   			for l in range(12,16):
   				atom=np.zeros(32)
   				atom[i]=1
   				atom[j]=1
   				atom[k]=1
   				atom[l]=1
   				data_exact_atoms_4by4.append(atom)
data_exact_atoms_4by4=np.array(data_exact_atoms_4by4)

#%%
#Here some useful functions are defined to deal with the atoms and plot them

def converting_explicit_list_to_binary_list(explicit_list, dim):
    res = np.zeros(2*(dim**2))
    res[explicit_list]=np.ones(len(explicit_list))
    return res

#receives an atom represented by a binary array 1 x (2n+1)
#where the first entry is for acknoledging if that atom is in p,
#the next n entries are for indicating the positions with black squares
#and the last n entries are for indicating the positions with white squares
def plot_atom(atom_as_list):
    #Converting the atom_as_list to atom_as_list_with_color_codes
    #grey=0.5 -> nothing there
    #white=0 
    #black=1
    #black&white=0.75
    dim = int( (len(atom_as_list)/2)**(1/2) )
    dim_squared = dim**2
    
    atom_as_list=list(atom_as_list)
    
    cells16_with_color = np.zeros(dim_squared) #atom_as_list[0:17].copy()
    for i in range(0,dim_squared):
        j=i+dim_squared
        if atom_as_list[i]==1 and atom_as_list[j]==1:
            cells16_with_color[i]=0.75
        if atom_as_list[i]==0 and atom_as_list[j]==1:
            cells16_with_color[i]=0
        if atom_as_list[i]==1 and atom_as_list[j]==0:
            cells16_with_color[i]=1
        if atom_as_list[i]==0 and atom_as_list[j]==0:
            cells16_with_color[i]=0.3
    
    res = np.reshape(cells16_with_color, (dim,dim), order='F')

    fig, ax = plt.subplots()
    plt.imshow(res, cmap="binary",vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    
    plt.show()
    return fig

#%%
#Approach to the problem using the Python-SAT Package 

#receives as input a set of atoms
#each atom is represented by a binary array 1 x (2n+1)
#where the first entry is for acknoledging if that atom is in p,
#the next n entries are for indicating the positions with black squares
#and the last n entries are for indicating the positions with white squares
def hitting_sets_of_the_set_of_atoms(data):

    h = Hitman(solver='m22', htype='lbx')
    
    #inputting the sets(atoms) to hit 
    for el in data:
        el_as_set = [i for i in range(len(el)) if el[i]==1]
        h.hit(el_as_set)
        
    #showing and blocking the hitting sets found
    set_of_hitting_sets=[]
    cur_set = h.get()
    while cur_set != None:
        set_of_hitting_sets.append(cur_set)
        h.block(cur_set)
        cur_set=h.get()
        
    set_of_hitting_sets.sort(key=len)
    
    return set_of_hitting_sets
  
#%%
#Approach to the problem using the STACCATO Algorithm
#Most of the names of variables are the same as in the article to make it easy to compare

def n11(set_of_atoms,e,j):
    #j is the given element(or constant)
    #for each atom we check if it contains the given element and if that atom is of interest
    #if that is the case we count that atom
    n = len([i for i in range(len(set_of_atoms.index)) if (set_of_atoms.iloc[i][str(j)]==1 and e[i]==1) ])
    #in the end it returns the number of atoms of interest that contain the constant j
    return n

def strip(set_of_atoms, e, j):
    A=set_of_atoms.copy()
    #if there is an atom of interest(small)(e=1) that is not in the constant j
    if len([i for i in range(len(A.index)) if (A.iloc[i][str(j)]==0 and e[i]==1) ])>0:
        rows_to_eliminate = [i for i in range(len(A.index)) if (A.iloc[i][str(j)]==1 and e[i]==1) ] 
        #getting the list of the labels of the rows to eliminate
        #we do this beacuse the method "drop" accepts as input the label of the rows we want to delete
        labels_of_rows_to_eliminate = [int(A.iloc[[r]].index[0]) for r in rows_to_eliminate]
        A = A.drop(labels_of_rows_to_eliminate, axis=0)
        e = np.delete(e, rows_to_eliminate, 0)
        #remove the column(this is the equivalent of STRIP_COMPONENT)
        A = A.drop([str(j)], axis=1)
    return A, e

def marking_small_atoms(set_of_atoms, k):
    e = np.zeros(len(set_of_atoms))
    for i in range(len(set_of_atoms)):
        if sum(set_of_atoms[i])<=k:
            e[i]=1
    return e

#Here we use the heuristic H
#ranking each constant by its relevance
def rank(set_of_atoms, e):
    coefs={}
    #for each element/column/constant we calculate the coefficient value
    for j in set_of_atoms.columns:
        #since all atoms provided are already small(all of them are atoms of interest)
        #we use only the n11 coefficient for this heuristic
        j=int(j)
        n11_ = n11(set_of_atoms,e,j)
        coefs[j]=n11_
    sorted_coefs = dict(sorted(coefs.items(), key=lambda x: x[1]))
    ranking = list(sorted_coefs.keys())
    #It returns the constants ordered by the number of atoms they take part 
    #starting from the least 
    return ranking

def staccato(set_of_atoms, e, seen, l, L):
    A = set_of_atoms
    #number of columns(elements) in A
    M=A.shape[1] 
    #Set of atoms in A that are of most interest (that have e=1)
    #The criteria of most interest is defined as the atoms with size smaller than a given k
    Tf = [A.iloc[u] for u in range(len(A.index)) if e[u]==1]
    #Computing the ranking of each element according to the number of atoms that are in that element 
    #(according to the formalization of AML)
    R = rank(A,e)
    #The minimal hitting set starts empty
    D=[]
    #Checking if there are elemnts that are in all subsets of interest
    elements_to_remove=[] #columns to be removed from A
    for j in A.columns:
        j=int(j)
        if n11(A,e,j) == len(Tf): #checks if element j is in all substes of interest
            elements_to_remove.append(str(j))
   
    D = [int(el) for el in elements_to_remove] 
    #Delete the columns(elements) of A
    A = A.drop(elements_to_remove, axis=1)
    
    #removing these elements from the ranking
    R = [el for el in R if str(el) not in elements_to_remove] 
    
    #Update the value fo seen
    if M>0:
        seen += len(elements_to_remove)*1/M
    
    #conditions to stop the recursion
    while (R!=[] and seen <= l and len(D)<=L):
        j = R.pop() 
        seen += 1/M
        A_ , e_ =  strip(A,e,j) 
        #Recursive Call
        D_ = staccato(A_, e_, seen, l, L) 
        #remove the hitting sets that are not minimal
        while D_ != [] :
            j_ = D_.pop() 
            if j_ not in D: 
                D.append(j_)
    return D

#%%
#Some functions to plot and evaluate the results from the previous approaches

def is_hit_set(set_of_atoms, candidate):
    trigger = False
    atoms_not_hit=[]
    for el in set_of_atoms:
        atom = set([i for i in range(len(el)) if el[i]==1])
        hit_set = set(candidate)
        #print(atom.intersection(min_hit_set))
        if atom.intersection(hit_set) == set():
            trigger = True
            atoms_not_hit.append(atom)
    if trigger: 
        return False, "atoms not hit: "+str(atoms_not_hit)
    return True

#plotting the sets returned by staccato using different values for lambda
#apparentely none of them is hitting set but almost
def plot_examples_returned_by_staccato(original_atoms, df, e, seen, L, dim):
    set_of_examples=[]
    set_of_examples_as_set=[]
    for l_ in range(10,80,10):
        l_ = l_/100
        candidate = staccato(df, e, seen, l_, L)
        print("lambda:", l_)
        print("Is this candidate a hitting set?")
        print(is_hit_set(original_atoms, candidate))
        example = converting_explicit_list_to_binary_list(candidate, dim)
        if set(candidate) not in set_of_examples_as_set:
            set_of_examples_as_set.append(set(candidate))
            set_of_examples.append(example)
            print("example_size:", sum(example))
            fig = plot_atom(example)
            #fig.savefig("3by3 L"+str(L)+" lambda"+str(l_)+".png", format="png", bbox_inches='tight', dpi=600)
            #fig.savefig("4by4 L"+str(L)+" lambda"+str(l_)+".png", format="png", bbox_inches='tight', dpi=600)
            #fig.savefig("37atoms_4by4 L"+str(L)+" lambda"+str(l_)+".png", format="png", bbox_inches='tight', dpi=600)

            
#plotting the sets returned by staccato using different values for lambda
#apparentely none of them is hitting set but almost
def plot_examples_returned_by_hitman(examples, dim):
    i=1
    for example in examples:
        example = converting_explicit_list_to_binary_list(example, dim)
        fig = plot_atom(example)
        #fig.savefig("3by3_"+str(i)+".png", format="png", bbox_inches='tight', dpi=600)
        #fig.savefig("4by4_"+str(i)+".png", format="png", bbox_inches='tight', dpi=600)
        #fig.savefig("37atoms_4by4_"+str(i)+".png", format="png", bbox_inches='tight', dpi=600)
        i+=1
#%%

#Here are the definitions of the DATAFRAMES that will be used to input in the functions
#We create dataframes so that each column has a label and we can manipulate the dataframe
#by the name of the columns and not by its indexes. If we did by the indexes with a np.array
#we would get into trouble once we deleted columns since the indexes would change

#Test atom to plot
test_el=np.array([ 1, 0, 0, 0,   0, 0, 0, 0,   0, 1, 0, 0,   0, 0, 0, 1, 
                   0, 0, 0, 0,   0, 0, 0, 0,   0, 1, 0, 1,   0, 0, 0, 0])    
#plot_atom(test_el)


#DataFrame for the 3by3 case with all the atoms
df1 = pd.DataFrame(data_exact_atoms_3by3, 
                   columns = [str(i) for i in range(data_exact_atoms_3by3.shape[1])])

e1=marking_small_atoms(data_exact_atoms_3by3, 3)

#DataFrame for the 4by4 case with 37 atoms coming of training(full-crossings):
df2 = pd.DataFrame(data_37_atoms_subset_of_the_model, 
                   columns = [str(i) for i in range(data_37_atoms_subset_of_the_model.shape[1])])
#We ignore the first element of each atom beacause it only means that it is in constant p
df2 = df2[df2.columns[1:]]

e2=marking_small_atoms(data_37_atoms_subset_of_the_model, 4)

#DataFrame for the 4by4 case with all the atoms
df3 = pd.DataFrame(data_exact_atoms_4by4, 
                   columns = [str(i) for i in range(data_exact_atoms_4by4.shape[1])])

e3=marking_small_atoms(data_exact_atoms_4by4, 4)

#The example in the paper
df99 = pd.DataFrame([[1,0,1],[0,1,1],[1,0,1]], columns=['1','2','3'])

e99 = marking_small_atoms([[1,0,1],[0,1,1],[1,0,1]], 3)
e99[2]=0
            
#%%

#Here are ready-to-run examples of executions of the previous solutions: staccato and hitman


#STACCATO

#plot_examples_returned_by_staccato(original_atoms,         df,  e, seen, L, dim)

#plot_examples_returned_by_staccato(data_exact_atoms_3by3,  df1, e1, 0,    3,  3)

#plot_examples_returned_by_staccato(data_exact_atoms_4by4, df3, e3, 0, 4, 4)

#plot_examples_returned_by_staccato(data_37_atoms_subset_of_the_model, df2, e2, 0, 6, 4)

#HITMAN

#examples1 = hitting_sets_of_the_set_of_atoms(data_exact_atoms_3by3)
#plot_examples_returned_by_hitman(examples1, 3)

#examples2 = hitting_sets_of_the_set_of_atoms(data_exact_atoms_4by4)
#plot_examples_returned_by_hitman(examples2,4)

#examples3 = hitting_sets_of_the_set_of_atoms(data_37_atoms_subset_of_the_model)
#examples3_size_less_than_6 = [el for el in examples3 if len(el)<6] 
#plot_examples_returned_by_hitman(examples3_size_less_than_6, 4)


