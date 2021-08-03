The file "./merged_res123457910_new.h5" contains the three-momenta of the tower constituents of jets with up to eight N-prongs.
To generate the low- and high-level datasets, type  

      python generate_datasets.py "./merged_res123457910_new.h5"

The file "generate_datasets.py" does the following:
1. Center the three-momenta of the towers according to the pT-weighted average in eta and phi. The circular average is used to calculate the average in phi.
2. The centered towers are saved in a ROOT TTree, which is used to calculate the Nsubjettiness variables. Note: this is done in the Delphes directory as all Nsubjettiness libraries are installed there, but no Delphes dependencies are needed.
