import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

# -- ROOT dependencies -------------------------------------------- #
import ROOT
import uproot
ROOT.gInterpreter.Declare('#include "./ttree_struct.cpp"')
from ROOT import JetBranch
# ----------------------------------------------------------------- #

def phi_in_range(phi):
    # make sure angle is in the range [-pi, pi]
    for i, p in enumerate(phi):
        if p < -np.pi:
            phi[i] += 2*np.pi
        if p > np.pi:
            phi[i] -= 2*np.pi
    return phi

def circular_mean(phi, weights):
    # calculat the circular mean of phi
    phi = phi_in_range(phi)
    x = 0
    y = 0
    for p, w in zip(phi, weights):
        x += np.cos(p) * w
        y += np.sin(p) * w
    return np.arctan2(y, x)

def fill_TTree(towers, TTree, jet_Pt, jet_Mass, targets):
    branch = JetBranch()
    TTree.Branch('jet_id', branch.jet_id, 'jet_id/I');
    TTree.Branch('jet_class', branch.jet_class, 'jet_class/I');
    TTree.Branch('jet_Pt', branch.jet_Pt, 'jet_Pt/F');
    TTree.Branch('jet_Mass', branch.jet_Mass, 'jet_Mass/F');
    TTree.Branch('towers_Pt', branch.towers_Pt, 'towers_Pt[230]/F');
    TTree.Branch('towers_Eta', branch.towers_Eta, 'towers_Eta[230]/F');
    TTree.Branch('towers_Phi', branch.towers_Phi, 'towers_Phi[230]/F');

    for i, threeM in enumerate(towers):
        branch.jet_id[0] = i
        branch.jet_class[0] = targets[i]
        branch.jet_Pt[0] = jet_Pt[i]
        branch.jet_Mass[0] = jet_Mass[i]
        for j in range(230):
            branch.towers_Pt[j] = threeM[j, 0]
            branch.towers_Eta[j] = threeM[j, 1]
            branch.towers_Phi[j] = threeM[j, 2]
        TTree.Fill()

def analyze_jets(input_file, output_h5file):
    inFile = h5py.File(input_file + ".h5", 'r')
    print("Reading file {}".format(input_file))

    print("Centering the towers...")
    towers = inFile['parsedTower'][()]  # raw towers
    for ijet, threeM in enumerate(towers):
        ix = np.where(threeM[:, 0] > 0)
        # normalize the pT
        threeM[:, 0][ix] /= np.sum(threeM[:, 0][ix])
        # center eta
        threeM[:, 1][ix] -= np.average(threeM[:, 1][ix],
                                       weights=threeM[:, 0][ix])
        # center phi
        threeM[:, 2][ix] -= circular_mean(threeM[:, 2][ix],
                                          weights=threeM[:, 0][ix])
        threeM[:, 2][ix] = phi_in_range(threeM[:, 2][ix])

    print("Saving towers in TTree...")
    # save the towers in a ROOT TTree
    inFileROOT = ROOT.TFile.Open(output_h5file + "_threeM.root", 'recreate');
    TTree = ROOT.TTree("constituents", "threeM");
    jet_Pt = inFile['HL'][()][:, -6]
    jet_Mass = inFile['HL'][()][:, -5]
    targets = inFile['target'][()]  # Nprong jet class
    fill_TTree(towers, TTree, jet_Pt, jet_Mass, targets)
    inFileROOT.Write();
    inFileROOT.Close();

    # -- calculate the Nsubs in the Delphes directory -- #
    cwd = os.getcwd()
    delphes_dir = "/home/alex/HEP_software/Delphes-3.4.2"
    execFile = "calculate_nsubs.cpp"
    ttreeFile = "ttree_struct.cpp"
    inFileDelphes = os.path.join(cwd, output_h5file + "_threeM.root")
    outFileDelphes = os.path.join(cwd, output_h5file + ".root")
    # copy all files to the Delphes dir
    cp_exec = "cp {} {}".format(execFile, delphes_dir)
    cp_ttree = "cp {} {}".format(ttreeFile, delphes_dir)
    cmnd_cp = "{} && {}".format(cp_exec, cp_ttree)
    os.system(cmnd_cp)
    # and run the ROOT macro
    print("Analyzing TTree on Delphes...")
    cmnd = "root -l -q " + execFile + "'(" + '"' + inFileDelphes + '"' + ',' + '"' + outFileDelphes +'"' + ")'"
    cmnd = "cd "+ delphes_dir +" && "+cmnd
    os.system(cmnd)

    # -- and lastly, save features in an h5 file -- #
    print("And saving to h5...")
    froot = uproot.open(outFileDelphes)
    target = froot['HL']['jet_class'].array()
    jet_PT = froot['HL']['jet_Pt'].array()
    jet_Mass = froot['HL']['jet_Mass'].array()
    jet_Eta = froot['HL']['jet_Eta'].array()
    jet_Phi = froot['HL']['jet_Phi'].array()
    towers_Pt = froot['HL']['towers_Pt'].array()
    towers_Eta = froot['HL']['towers_Eta'].array()
    towers_Phi = froot['HL']['towers_Phi'].array()
    nsubs_b05 = froot['HL']['nsubs_b05'].array()
    nsubs_b10 = froot['HL']['nsubs_b10'].array()
    nsubs_b20 = froot['HL']['nsubs_b20'].array()
    constituents_threeM = np.stack((towers_Pt, towers_Eta, towers_Phi), axis=-1)

    fh5 = h5py.File(output_h5file + ".h5", 'a')
    fh5.create_dataset("target", data=target)
    fh5.create_dataset("jet_PT", data=jet_PT)
    fh5.create_dataset("jet_Mass", data=jet_Mass)
    fh5.create_dataset("jet_Eta", data=jet_Eta)
    fh5.create_dataset("jet_Phi", data=jet_Phi)
    fh5.create_group("Constituents")
    fh5.create_dataset("Constituents/threeM", data=constituents_threeM)
    fh5.create_group("Nsubs")
    fh5.create_dataset("Nsubs/Nsubs_beta05", data=nsubs_b05)
    fh5.create_dataset("Nsubs/Nsubs_beta10", data=nsubs_b10)
    fh5.create_dataset("Nsubs/Nsubs_beta20", data=nsubs_b20)
    fh5.close()
    print("done :)")

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5file", type=str,
                        help='File containing towers three-momenta -- no ext')
    parser.add_argument("output_h5file", type=str,
                        help='File to save the dataset to -- no ext')
    args = parser.parse_args()
    analyze_jets(args.input_h5file, args.output_h5file)
