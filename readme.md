## DeepMTC：Predict protein multi-label subcellular localization and function

This is the repository related to our manuscript Deep learning model for protein multi-label subcellular localization and function prediction based on multi-task collaborative training, currently in submission at Knowledge-based Systems.

## Code
### Environment Requirement
The code has been tested running under Python 3.8.16. The required packages are as follows:
- numpy == 1.23.5
- numpy-base == 1.23.5
- openfold == 1.0.0
- networkx == 3.1
- scipy == 1.10.1
- pytorch == 1.13.1
- pytorch-lightning == 1.5.10
- pytorch-cuda ==11.6

## Files

1. Dataset: 
           goa_human_isoform.gaf and goa_mouse_isoform.gaf store the protein sequence GO annotation information;   
           idmapping_2024_03_19 (1).xlsx store the protein multi-label subcellular localization; 
           HM.fasta store the protein sequence.

2. model_para: 
           save the optimal parameters of the model

3. src:        
           a.Main_model.py：the DeepMTC framework； 
           b.GT.py: the graph transformer block;
           c.Fun_attention3.py: functional cross-attention block;
           d.Multitask_col_train.py: the training model preserves the optimal parameters of the model and is tested on an independent test set.



 

