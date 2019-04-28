README:  Description of files in Extended Data, divided by subfolder.


figure_models:

    figure_1:
        "top_L2_epi_pairs_on_2gb1.pse":  top L/2 longrange epistasis pairs connected with lines on the experimental structure 2gb1

    figure_4:
        "gb1_model_vs_experimental_structure_view1--ribbon.pse":  the final folded model of gb1 versus the experimental structure 2gb1
        "gb1_model_vs_experimental_structure_view2--ribbon.pse":  same as above, but rotated 180 degrees
        "ww_model_vs_experimental_structure_view1--ribbon.pse":  the final folded model of ww versus the experimental structure 1jmq
        "ww_model_vs_experimental_structure_view2--ribbon.pse":  same as above, but rotated 180 degrees

    figure_5:
        "any_random_5perc_ensemble.pse":  final folded models (n=10) from unguided mutant libraries 5% the size of the full dataset, aligned to 2gb1
        "one_del_5perc_ensemble.pse":  final folded models (n=10) from partially guided mutant libraries 5% the size of the full dataset, aligned to 2gb1
        "both_del_5perc_ensemble.pse":  final folded models (n=10) from pairwise guided mutant libraries 5% the size of the full dataset, aligned to 2gb1


    supplementary_figure 7:
        "gb1_FPs_at_ligand_interface.pse":  gb1 bound to IgG with residues enriched in false-positives shown as spheres, 1fcc
        "ww_FPs_at_ligand_interface.pse":  ww bound to peptide ligand with residues enriched in false-positives shown as spheres, 1jmq
        "rrm_FPs_at_ligand_interface.pse":   rrm bound to RNA ligand with residues enriched in false-positives shown as spheres, 1cvj



folding directory:

    gb1_final_models:  the two PDB files for the predicted models that are reported in the main text
        "GB1_best_in_top_25_scoring.pdb":  The model folded using predicted hydrogen bonds
        "GB1_no_betas_best_in_top_25_scoring.pdb":  The model folded without predicting hydrogen bonds
        
    ww_final_model:  
        "WW_best_in_top_25_scoring.pdb":  The model reported in the main text
        
    codes_and_inputs:
        
        cns_input: input files to the folding program CNS
        
            CNS-format *.tbl files:
            
            "gb1_56_distance_couplings.tbl":  an example file for GB1 used to input 56 epistatic pairs as distance constraints
            "gb1_psipred*_hbonds_*.tbl":  The hydrogen bonds (between N and O atoms) that were added in as additional constraints for gb1, 
            based on 3 different secondary structure ranges used.  Note that since the register is still ambiguous, we have two hydrogen bond 
            sets for each secondary structure.  We use them in separate runs, and then pool models together for ranking.
            "gb1_ss_angle.tbl":  An example of the dihedral angle constraints based on secondary structure used for GB1
            "gb1_ss_distance.tbl":  An example of the secondary structure distance constraints used for GB1
            "WW_39_couplings.tbl":  an example file for WW used to input 39 epistatic pair distance constraints.
            "ww_hbond_psipred1_*.tbl":  The hydrogen bonds (between N and O atoms) that were added in as additional constraints for WW
            "WW_ss_angle.tbl":  The dihedral angle constraints based on secondary structure used for WW
            "WW_ss_distance.tbl":  The distance constraints based just on secondary structure that were used for WW
            
            CNS-format *.inp file:
            
            "sample_DGSA_betas.inp":  The generic .inp script that could be used to run distance geometry-simulated annealing for our folding     
            runs.  Values in '{{ __ }}' can be replaced programmatically by the appropriate files.
            
        secondary_structure:  Comma-separated values files that show the secondary structure ranges used for GB1 and WW
        
            "gb1_psipred*_secondary_structure.csv":  The three secondary structure ranges used for GB1
            "ww_secondary_structure_psipred1.csv":  The PSIPRED secondary structure range for WW, formatted to run in our pipeline
            
        code:  Main python code used to predict hydrogen bonds, and then fold with the output files.  Note that both require the evcouplings package (available at <https://github.com/debbiemarkslab/  EVcouplings>) to be installed in your python directory.
            
            "beta_prediction.py":  Provides two output *.tbl files (with same filename prefix as your input file with epistatic constraints) with CNS-formatted constraints between N and O atoms, predicted from a *.csv file containing epistatic pairs.  If run at the command line, it takes two input arguments:  1) the filename containing the epistatic contacts, minimally containing columns named 'i','j', and 'cn' (the epistatic score between positions i and j in the sequence), and 2) the filename with the secondary structure ranges, containing columns 'i' (sequence position) and 'sec_struct_3state', which denotes whether the residue at that position is H (helix), E (strand), or C (coil).
 

subsampling directory:

    constraints:  top epistatic pairs from subsampled libraries used to fold

        "any_pair.zip":  epistatic pairs from unguided libraries, drawn 10 times each for 0.5%-50% of the full dataset

        "one_del.zip":  epistatic pairs from partially guided libraries (>=1 deleterious mutant per pair), drawn 10 times each for 0.5%-25% of the full dataset

        "del_pair.zip":  epistatic pairs from fully guided libraries (pairs of 2 deleterious mutants), drawn 10 times each for 0.5%-25% of the full dataset
        
    
        