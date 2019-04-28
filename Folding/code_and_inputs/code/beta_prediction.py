import pandas as pd
import numpy as np

from evcouplings.fold import run_cns, cns_dgsa_inp, cns_dist_restraint, ec_dist_restraints, cns_dist_restraint
from evcouplings.fold.restraints import _folding_config
from evcouplings.compare import (
    ClassicPDB, PDB, DistanceMap, SIFTS, intra_dists,
    multimer_dists, coupling_scores_compared
)

from evcouplings.fold import maxcluster_clustering_table, dihedral_ranking, compare_models_maxcluster
from evcouplings.couplings import add_mixture_probability
from evcouplings.visualize import plot_contact_map, plot_context, pymol_pair_lines

import sys

def get_strand_pairings(list_beta_pairs,ec_file,init_struct=False,chain_id = ' '):
    
    #Inputs:
    #list_beta_pairs:  e.g. [(1,10),(23,43)]
    #ecs:  pandas Dataframe with ecs
    
    # Load in ecs file
    ecs = pd.read_csv(ec_file)
    
    #max_linker = 10
    max_linker = 6
    #ec_cutoff = ecs.iloc[2*max(ecs['j'])]['cn']
    ec_cutoff = ecs.iloc[min(len(ecs),(max(ecs['j'])-min(ecs['i'])+1))]['cn']
    
    num_betas = len(list_beta_pairs)
    
    # Create list of pairs that can be updated
    strand_pairs = {}
    
    # Sort beta strands in case they're out of order
    list_beta_pairs = sorted(list_beta_pairs, key=lambda x: x[0])
    
    # Load in our predicted initial structure and get a dataframe with distance coords for each pairing
    if init_struct:
        coords = ClassicPDB.from_file(init_struct).get_chain(chain_id)
        dist_map = DistanceMap.from_coords(coords)
        ecs_with_dist = coupling_scores_compared(ecs,dist_map)
    
    # Check if any strands are too close together and must be coupled
    #length_linkers = []
    for i in range(num_betas-1):
        
        length_linker = list_beta_pairs[i+1][0] - list_beta_pairs[i][1] - 1
        if length_linker < max_linker:
            #print('Linker: {} to {}'.format(i+1,i))
            if i in strand_pairs:
                strand_pairs[i] = strand_pairs[i]+[(i+1,('antiparallel','linker'))]
            else:
                strand_pairs[i] = [(i+1,('antiparallel','linker'))]
            #strand_pairs[i] = [(i+1,('antiparallel','linker'))]
            if (i+1) in strand_pairs:
                strand_pairs[i+1] = strand_pairs[i+1] + [(i,('antiparallel','linker'))]
            else:
                strand_pairs[i+1] = [(i,('antiparallel','linker'))]
            #strand_pairs[i+1] = [(i,('antiparallel','linker'))]
    #print(strand_pairs)
    
    # Now that we've eliminated clashes based on linker length, let's go through top ECs 
    # and choose strand pairings on basis of top EC
    for i in range(num_betas):
        
        # Creating temporary lists of top ec scores, and the strand index the EC forms a pair with
        top_ec_score_list = []
        indices = []
        ec_pairs = []
        
        # Go through each potential partner strand
        for j in range(num_betas):
            
            # Don't compare same-strand ecs
            if j==i:
                continue
                
            indices.append(j)
            b1_start = list_beta_pairs[i][0]
            b1_end = list_beta_pairs[i][1]
            b2_start = list_beta_pairs[j][0]
            b2_end = list_beta_pairs[j][1]          
            
            # Get ECs between our strand of interest and its current partner strand, and append top scoring EC to list
            temp_list = ecs.query('(i >= @b1_start and i <= @b1_end and j >= @b2_start and j<= @b2_end) or (j >= @b1_start and j <= @b1_end and i >= @b2_start and i<=@b2_end)')
            if len(temp_list)>0:
            	top_ec = temp_list.iloc[0]['cn']
            	top_ec_i = temp_list.iloc[0]['i']
            	top_ec_j = temp_list.iloc[0]['j']
            	
            else:
            	top_ec = 0
            	top_ec_j = 0
            	top_ec_i = 0
            #top_ec = ecs.query('(i >= @b1_start and i <= @b1_end and j >= @b2_start and j<= @b2_end) or (j >= @b1_start and j <= @b1_end and i >= @b2_start and i<=@b2_end)').iloc[0]
            
            #cn = top_ec['cn']
            cn = top_ec
            top_ec_score_list.append(cn)
            ec_pairs.append((top_ec_i,top_ec_j))
            
        #print(top_ec_score_list)    
        
        # Get strand identifier and EC score for best and second-best strand-strand pairings
        best_strand = indices[np.argmax(top_ec_score_list)]
        max_ec = max(top_ec_score_list)
        
        sbi = [top_ec_score_list.index(x) for x in sorted(top_ec_score_list, reverse=True)][1]        
        second_best_strand = indices[sbi]
        second_best_ec = [x for x in sorted(top_ec_score_list, reverse=True)][1]
        
        # Guess whether parallel or antiparallel based on surrounding pair scores
        def _check_parallel(ec_pair,ecs,list_beta_pairs,istrand,jstrand):
            b1_start = list_beta_pairs[istrand][0]
            b1_end = list_beta_pairs[istrand][1]
            b2_start = list_beta_pairs[jstrand][0]
            b2_end = list_beta_pairs[jstrand][1] 
            
            i_val = ec_pair[0]
            j_val = ec_pair[1]
            
            #print(str(list_beta_pairs[istrand])+' '+str(list_beta_pairs[jstrand]))
            
            strand_ecs = ecs.query('(i>=@b1_start and i<=@b1_end and i!=@i_val and j>=@b2_start and j<=@b2_end and j!=@j_val) or (j>=@b1_start and j<=@b1_end and j!=@j_val and i>=@b2_start and i<=@b2_end and i!=@i_val )').sort_values(by='cn',ascending=False)
            if len(strand_ecs) < 1:
            	#print('not enough ecs')
            	#print(strand_ecs)
            	return 'not enough info to determine strand orientation'
            new_i = strand_ecs.iloc[0]['i']
            new_j = strand_ecs.iloc[0]['j']
            
            if (i_val - new_i)*(j_val - new_j) < 0:
            	return 'antiparallel'
            elif (i_val - new_i)*(j_val - new_j) > 0:
            	return 'parallel'
            else:
            	print(strand_ecs)
            	return 'Error: dx 0'
            # TODO:  add check if this goes beyond length of protein sequence
            
            #antiparallel_score = float(ecs.query('i==(@i_val-1) and j==(@j_val+1)').iloc[0]['cn'])+float(ecs.query('i==(@i_val+1) and j==(@j_val-1)').iloc[0]['cn'])
            #parallel_score = float(ecs.query('i==(@i_val-1) and j==(@j_val-1)')['cn'])+float(ecs.query('i==(@i_val+1) and j==(@j_val+1)')['cn'])
            
            #if parallel_score > antiparallel_score:
            #    return 'parallel'
            #else:
            #    return 'antiparallel'
        
        #best_strand_orientation = _check_parallel(ec_pairs[np.argmax(top_ec_score_list)],ecs)
        #second_best_strand_orientation =_check_parallel(ec_pairs[sbi],ecs)
        best_strand_orientation = _check_parallel(ec_pairs[np.argmax(top_ec_score_list)],ecs,list_beta_pairs,i,best_strand)
        second_best_strand_orientation =_check_parallel(ec_pairs[sbi],ecs,list_beta_pairs,i,second_best_strand)
        
        # If EC score is below cutoff, don't include
        if max_ec < ec_cutoff:
            max_ec = 'below_cutoff'
        if second_best_ec < ec_cutoff:
            second_best_ec = 'below_cutoff'
            
        # If we haven't already added the strand based on linker constraints, put in both best and second-best pairings
        if i not in strand_pairs:
            strand_pairs[i] = [(best_strand,(best_strand_orientation,max_ec)),(second_best_strand,(second_best_strand_orientation,second_best_ec))]
        
        # If we already have a strand-strand pairing based on linker constraints, put in the pairing suggested by 
        #either best or second-best EC based on which one is already present
        elif len(strand_pairs[i]) == 1:
            
            if strand_pairs[i][0][0] == best_strand:
                strand_pairs[i] = strand_pairs[i]+[(second_best_strand,(second_best_strand_orientation,second_best_ec))]
            else:
                strand_pairs[i] = strand_pairs[i]+[(best_strand,(best_strand_orientation,max_ec))]
                
        # If we've already identified two partners for this strand on the basis of linker, don't add anything else
        else:
            continue
        
    
    # prune list to make sure we don't have any one-directional pairings, e.g. 1->3 but not 3->1
    for i in range(len(strand_pairs)):
    	
    	s1 = strand_pairs[i][0][0]
    	s2 = strand_pairs[i][1][0]
    	
    	if i not in [strand_pairs[s1][0][0],strand_pairs[s1][1][0]]:
    		#print('{}: in {} or {}?'.format(i,strand_pairs[s1][0][0],strand_pairs[s1][0][1]))
    		strand_pairs[i][0] = (strand_pairs[i][0][0],(strand_pairs[i][0][1][0],'one_directional'))
    	if i not in [strand_pairs[s2][0][0],strand_pairs[s2][1][0]]:
    		strand_pairs[i][1] = (strand_pairs[i][1][0],(strand_pairs[i][1][1][0],'one_directional'))
    		     
    return strand_pairs, list_beta_pairs 
    
def get_hydrogen_bonds(strand_pairings,list_beta_pairs,ec_file):
    
    ecs = pd.read_csv(ec_file)
    num_strands = len(strand_pairings)
    
    pattern1 = {}
    pattern2 = {}
    
    def _check_value_in_tuplelist(val,tuple_list):
        if [item for item in tuple_list if val in item]:
            return True
        else:
            return False
        
    def _check_ij_in_range(r,i,j):
        if (i >= min(r) and i <= max(r)):
            return i
        elif (j >= min(r) and j <= max(r)):
            return j
        
    def _range_length_antiparallel(r1,r2,i,j):
        
        return min(abs(i-min(r1)),abs(j-max(r2))) + min(abs(i-max(r1)),abs(j-min(r2))) + 1
    
    def _range_length_parallel(r1,r2,i,j):
        
        return min(abs(i-min(r1)),abs(j-min(r2))) + min(abs(i-max(r1)),abs(j-max(r2))) + 1
    
    def _add_hbonds_based_strand_pairings(b1_range,b2_range,b2_orientation,ecs,pattern1,pattern2):
        
        b1_start = b1_range[0]
        b1_end = b1_range[1]
        b2_start = b2_range[0]
        b2_end = b2_range[1]
        
        print('Checking {}-{} against {}-{}'.format(b1_start,b1_end,b2_start,b2_end))
        
        # Get top ECs
        top_ecs = ecs.query('(i >= @b1_start and i <= @b1_end and j >= @b2_start and j<= @b2_end) or (j >= @b1_start and j <= @b1_end and i >= @b2_start and i<=@b2_end)')
        current_range_length = 0
        optimal_range_length = min(abs(b1_end - b1_start)+1, abs(b2_end-b2_start)+1)
        current_ec_index = 0
        ival = 0
        jval = 0
        hbond1 = []
        hbond2 = []
                        
        pattern1_list = [k[0] for k in pattern1.keys()] + [k[1] for k in pattern1.keys()]
        pattern2_list = [k[0] for k in pattern2.keys()] + [k[1] for k in pattern2.keys()]
        
        if b2_orientation == 'antiparallel':
                            
            # Get the highest EC between the two strands that maximizes the possible range of paired amino acids
            #while current_range_length != optimal_range_length:
            while (current_range_length != optimal_range_length) and (current_ec_index < len(top_ecs)):
                    
                ival0 = top_ecs.iloc[current_ec_index]['i']
                jval0 = top_ecs.iloc[current_ec_index]['j']
                    
                ival = _check_ij_in_range((b1_start,b1_end),ival0,jval0)
                jval = _check_ij_in_range((b2_start,b2_end),ival0,jval0)
                    
                current_range_length = _range_length_antiparallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
                hbond1,hbond2 = _get_hbond_pairs_antiparallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
                current_ec_index = current_ec_index+1
                
            if (current_ec_index == len(top_ecs)) and (len(top_ecs)>0):
                print(top_ecs)
                current_ec_index = 0
                ival0 = top_ecs.iloc[current_ec_index]['i']
                jval0 = top_ecs.iloc[current_ec_index]['j']
                    
                ival = _check_ij_in_range((b1_start,b1_end),ival0,jval0)
                jval = _check_ij_in_range((b2_start,b2_end),ival0,jval0)
                    
                current_range_length = _range_length_antiparallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
                hbond1,hbond2 = _get_hbond_pairs_antiparallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
            
            # If antiparallel by linker but no ECs, just assume b1_end goes with b2_start    
            elif len(top_ecs)==0:
            	current_ec_index = 0
            	current_range_length = _range_length_antiparallel((b1_start,b1_end),(b2_start,b2_end),b1_end,b2_start)
            	hbond1,hbond2 = _get_hbond_pairs_antiparallel((b1_start,b1_end),(b2_start,b2_end),b1_end,b2_start)
                
            
            print('EC index used: {}'.format(current_ec_index))
            
            if (not any([_check_value_in_tuplelist(x,hbond1) for x in pattern1_list])) and (not any([_check_value_in_tuplelist(x,hbond2) for x in pattern2_list])):
                for entry in hbond1:
                    pattern1[entry] = 'antiparallel'
                for entry in hbond2:
                    pattern2[entry] = 'antiparallel'
            elif (not any([_check_value_in_tuplelist(x,hbond1) for x in pattern2_list])) and (not any([_check_value_in_tuplelist(x,hbond2) for x in pattern1_list])):
                for entry in hbond1:
                    pattern2[entry] = 'antiparallel'
                for entry in hbond2:
                    pattern1[entry] = 'antiparallel'
            else:
                print('Error:  amino acid projected to have multiple hbonds')

        if b2_orientation == 'parallel':
        
            #print('in parallel')
                            
            # Get the highest EC between the two strands that maximizes the possible range of paired amino acids
            while (current_range_length != optimal_range_length) and (current_ec_index < len(top_ecs)):
                    
                ival0 = top_ecs.iloc[current_ec_index]['i']
                jval0 = top_ecs.iloc[current_ec_index]['j']
                    
                ival = _check_ij_in_range((b1_start,b1_end),ival0,jval0)
                jval = _check_ij_in_range((b2_start,b2_end),ival0,jval0)
                    
                current_range_length = _range_length_parallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
                hbond1_NO,hbond1_ON,hbond2_NO,hbond2_ON = _get_hbond_pairs_parallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
                current_ec_index = current_ec_index+1
                
            if (current_ec_index == len(top_ecs)) and (len(top_ecs)>0):
                current_ec_index = 0
                ival0 = top_ecs.iloc[current_ec_index]['i']
                jval0 = top_ecs.iloc[current_ec_index]['j']
                    
                ival = _check_ij_in_range((b1_start,b1_end),ival0,jval0)
                jval = _check_ij_in_range((b2_start,b2_end),ival0,jval0)
                    
                current_range_length = _range_length_parallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
                hbond1_NO,hbond1_ON,hbond2_NO,hbond2_ON = _get_hbond_pairs_parallel((b1_start,b1_end),(b2_start,b2_end),ival,jval)
                
            
            hbond1 = hbond1_NO + hbond1_ON
            hbond2 = hbond2_NO + hbond2_ON
            
            print(hbond1)
            print(hbond2)
                        
            print('EC index used: {}'.format(current_ec_index))
            
            if (not any([_check_value_in_tuplelist(x,hbond1) for x in pattern1_list])) and (not any([_check_value_in_tuplelist(x,hbond2) for x in pattern2_list])):
                for entry in hbond1:
                    pattern1[entry] = 'parallel'
                for entry in hbond2:
                    pattern2[entry] = 'parallel'
            elif (not any([_check_value_in_tuplelist(x,hbond1) for x in pattern2_list])) and (not any([_check_value_in_tuplelist(x,hbond2) for x in pattern1_list])):
                for entry in hbond1:
                    pattern2[entry] = 'parallel'
                for entry in hbond2:
                    pattern1[entry] = 'parallel'
            else:
                print('Error:  amino acid projected to have multiple hbonds')
                #print(hbond2)
            
            
            #if (not any([_check_value_in_tuplelist(x,hbond1) for x in pattern1_list])) and (not any([_check_value_in_tuplelist(x,hbond2) for x in pattern2_list])):
            #    for entry in hbond1:
            #        pattern1[entry] = 'parallel'
            #    for entry in hbond2:
            #        pattern2[entry] = 'parallel'
            #elif (not any([_check_value_in_tuplelist(x,hbond1) for x in pattern2_list])) and (not any([_check_value_in_tuplelist(x,hbond2) for x in pattern1_list])):
            #    for entry in hbond1:
            #        pattern2[entry] = 'parallel'
            #    for entry in hbond2:
            #        pattern1[entry] = 'parallel'
            #else:
            #    print('Error:  amino acid projected to have multiple hbonds')
        
 
        return pattern1,pattern2
    
    def _get_hbond_pairs_antiparallel(r1,r2,i,j):
        
        subrange_dx1 = min(abs(i-min(r1)),abs(j-max(r2)))
        subrange_dx2 = min(abs(i-max(r1)),abs(j-min(r2)))
        
        b1_top = i-subrange_dx1
        b1_bottom = i+subrange_dx2
        b2_top = j+subrange_dx1
        b2_bottom = j-subrange_dx2
        
        #hbond1 = [(x,y) for x,y in zip(range(b1_top,b1_bottom+1,2),range(b2_top,b2_bottom-1,-2))]
        #hbond2 = [(x,y) for x,y in zip(range(b1_top+1,b1_bottom+1,2),range(b2_top-1,b2_bottom-1,-2))]
        
        hbond1 = [('N'+str(x),'O'+str(y)) for x,y in zip(range(b1_top,b1_bottom+1,2),range(b2_top,b2_bottom-1,-2))] + [('O'+str(x),'N'+str(y)) for x,y in zip(range(b1_top,b1_bottom+1,2),range(b2_top,b2_bottom-1,-2))]
        hbond2 = [('N'+str(x),'O'+str(y)) for x,y in zip(range(b1_top+1,b1_bottom+1,2),range(b2_top-1,b2_bottom-1,-2))] + [('O'+str(x),'N'+str(y)) for x,y in zip(range(b1_top+1,b1_bottom+1,2),range(b2_top-1,b2_bottom-1,-2))]

        
        return hbond1,hbond2
    
    def _get_hbond_pairs_parallel(r1,r2,i,j):
        
        subrange_dx1 = min(abs(i-min(r1)),abs(j-min(r2)))
        subrange_dx2 = min(abs(i-max(r1)),abs(j-max(r2)))
        
        b1_bottom = i-subrange_dx1
        b1_top = i+subrange_dx2
        b2_top = j+subrange_dx2
        b2_bottom = j-subrange_dx1
        
        #hbond1 = [(x,y) for x,y in zip(range(b1_bottom,b1_top+1,2),range(b2_bottom,b2_top+1,2))]
        #hbond2 = [(x,y) for x,y in zip(range(b1_bottom+1,b1_top+1,2),range(b2_bottom+1,b2_top+1,2))]
        
        hbond1_NO = [('N'+str(x),'O'+str(y)) for x,y in zip(range(b1_bottom,b1_top+1,2),range(b2_bottom-1,b2_top,2))]
        hbond1_ON = [('O'+str(x),'N'+str(y)) for x,y in zip(range(b1_bottom,b1_top+1,2),range(b2_bottom+1,b2_top+2,2))]
        hbond2_NO = [('N'+str(x),'O'+str(y)) for x,y in zip(range(b2_bottom,b2_top+1,2),range(b1_bottom-1,b1_top,2))]
        hbond2_ON = [('O'+str(x),'N'+str(y)) for x,y in zip(range(b2_bottom,b2_top+1,2),range(b1_bottom+1,b1_top+2,2))]
        
        #hbond1_NO = [('N'+str(x),'O'+str(y)) for x,y in zip(range(b1_bottom+1,b1_top+1,2),range(b2_bottom,b2_top,2))]
        #hbond1_ON = [('O'+str(x),'N'+str(y)) for x,y in zip(range(b1_bottom+1,b1_top,2),range(b2_bottom+2,b2_top+1,2))]
        #hbond2_NO = [('N'+str(x),'O'+str(y)) for x,y in zip(range(b2_bottom+1,b2_top+1,2),range(b2_bottom,b1_top,2))]
        #hbond2_ON = [('O'+str(x),'N'+str(y)) for x,y in zip(range(b2_bottom+1,b2_top,2),range(b1_bottom+2,b1_top+1,2))]
        
        hbond1_NO = [x for x in hbond1_NO if (((int(x[0][1:]) >= r1[0]) and (int(x[0][1:]) <= r1[1])) and ((int(x[1][1:]) >= r2[0]) and (int(x[1][1:]) <= r2[1])))]
        hbond1_ON = [x for x in hbond1_ON if (((int(x[0][1:]) >= r1[0]) and (int(x[0][1:]) <= r1[1])) and ((int(x[1][1:]) >= r2[0]) and (int(x[1][1:]) <= r2[1])))]
        
        hbond2_NO = [x for x in hbond2_NO if (((int(x[0][1:]) >= r2[0]) and (int(x[0][1:]) <= r2[1])) and ((int(x[1][1:]) >= r1[0]) and (int(x[1][1:]) <= r1[1])))]
        hbond2_ON = [x for x in hbond2_ON if (((int(x[0][1:]) >= r2[0]) and (int(x[0][1:]) <= r2[1])) and ((int(x[1][1:]) >= r1[0]) and (int(x[1][1:]) <= r1[1])))]
        
        return hbond1_NO,hbond1_ON,hbond2_NO,hbond2_ON
        
        
    for i in range(num_strands):
        b1_start = list_beta_pairs[i][0]
        b1_end = list_beta_pairs[i][1]
        
        print(strand_pairings)
        b2 = strand_pairings[i][0][0]
        b2_start = list_beta_pairs[b2][0]
        b2_end = list_beta_pairs[b2][1]
        b2_orientation = strand_pairings[i][0][1][0]
        b2_cutoff = strand_pairings[i][0][1][1]
        
        b3 = strand_pairings[i][1][0]
        b3_start = list_beta_pairs[b3][0]
        b3_end = list_beta_pairs[b3][1]
        b3_orientation = strand_pairings[i][1][1][0]
        b3_cutoff = strand_pairings[i][1][1][1]
        
        # Don't include pairings we've already done or below-cutoff ones
        #print(b2_cutoff)
        #print(b3_cutoff)
        if (b2 > i) and (b2_cutoff != 'below_cutoff') and (b2_cutoff != 'one_directional'):
            #print(b2_cutoff)        
            pattern1,pattern2 =_add_hbonds_based_strand_pairings((b1_start,b1_end),(b2_start,b2_end),b2_orientation,ecs,pattern1,pattern2)
             
        if (b3 > i) and (b3_cutoff != 'below_cutoff') and (b3_cutoff != 'one_directional'):
            #print(b3_orientation)
            pattern1,pattern2 = _add_hbonds_based_strand_pairings((b1_start,b1_end),(b3_start,b3_end),b3_orientation,ecs,pattern1,pattern2)

    return pattern1, pattern2
    
def write_hbonds_cns_command(patternx,filename):
    with open(filename,'w') as f:
        for k,v in patternx.items(): 
            f.write('assign (resid {} and name {}) (resid {} and name {})  3 0.5 0.5 ! beta\n'.format(k[0][1:],k[0][0],k[1][1:],k[1][0]))
                         
            #if v == 'antiparallel':
            #    f.write('assign (resid {} and name O) (resid {} and name N)  2 0.5 0.5 ! beta\n'.format(k[0],k[1]))
            #    f.write('assign (resid {} and name N) (resid {} and name O)  2 0.5 0.5 ! beta\n'.format(k[0],k[1]))
    #return True

def get_sec_struct_from_file(filename):
    
    df = pd.read_csv(filename)
    
    results = []
    curr = 0
    in_beta = False
    beta_start = 0
    beta_end = 0
    
    for i in range(len(df)):
        
        if in_beta == False:
            if df.iloc[i]['sec_struct_3state'] == 'E':
                in_beta = True
                beta_start = df.iloc[i]['i']
        else:
            if df.iloc[i]['sec_struct_3state'] != 'E':
                beta_end = df.iloc[i-1]['i']
                results.append((beta_start,beta_end))
                in_beta = False
    if in_beta:
        results.append((beta_start,df.iloc[-1]['i']))
        
    return results
    
def run_folding_with_betas(ecfile,sp=[(229,235),(238,246),(268,272),(276,282)],secstruct_file = '',prefix=''):
	#print('in main')   
	
	if len(secstruct_file)>0:
	    sp = get_sec_struct_from_file(secstruct_file)
	    
	print(sp)
	sp,lbp = get_strand_pairings(sp,ecfile)
	print(sp)
	
	if prefix=='':
	    prefix = '.'.join(ecfile.split('.')[:-1])
	    
	p1,p2 = get_hydrogen_bonds(sp,lbp,ecfile)
	print(p1)
	print(p2)
	write_hbonds_cns_command(p1,prefix+'_hbonds_1.tbl')
	write_hbonds_cns_command(p2,prefix+'_hbonds_2.tbl')
	
	return prefix+'_hbonds_1.tbl',prefix+'_hbonds_2.tbl'

if __name__ == "__main__":
	
	
	ec_file = sys.argv[1]
	ss_file = sys.argv[2]

	org_sp = get_sec_struct_from_file(ss_file)
	    
	sp,lbp = get_strand_pairings(org_sp,ec_file)
	
	prefix = '.'.join(sys.argv[1].split('.')[:-1])
	p1,p2 = get_hydrogen_bonds(sp,lbp,sys.argv[1])
	
	write_hbonds_cns_command(p1,prefix+'_hbonds_1.tbl')
	write_hbonds_cns_command(p2,prefix+'_hbonds_2.tbl')