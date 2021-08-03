#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#endif

#include <stdio.h>
#include <iostream>

#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"
//#include "fastjet/contribs/EnergyCorrelator/EnergyCorrelator.hh"
#include "fastjet/contribs/Nsubjettiness/Nsubjettiness.hh"
#include "fastjet/contribs/Nsubjettiness/Njettiness.hh"
#include "fastjet/contribs/Nsubjettiness/NjettinessPlugin.hh"
#include "fastjet/contribs/Nsubjettiness/ExtraRecombiners.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Recluster.hh"

#include "ttree_struct.cpp"

using namespace std;
using namespace fastjet;
using namespace fastjet::contrib;

fastjet::JetDefinition *jet_def = new fastjet::JetDefinition(fastjet::antikt_algorithm, 1.2);

Double_t Nsubjettiness_Nbeta(PseudoJet Jet, int N, float beta)
{
    Nsubjettiness nSub(N, KT_Axes(), UnnormalizedMeasure(beta));
    return nSub(Jet);
}

void analyzeTree(TTree *inputTree, TTree *outputTree)
{
  // read input tree
  // -------------------------------------------------------------------------//
  Int_t   jet_id[1], jet_class[1];
  Float_t jet_Pt[1], jet_Mass[1];
  Float_t towers_Pt[230], towers_Eta[230], towers_Phi[230];

  inputTree->SetBranchAddress("jet_id", &jet_id);
  inputTree->SetBranchAddress("jet_class", &jet_class);
  inputTree->SetBranchAddress("jet_Pt", &jet_Pt);
  inputTree->SetBranchAddress("jet_Mass", &jet_Mass);
  inputTree->SetBranchAddress("towers_Pt", &towers_Pt);
  inputTree->SetBranchAddress("towers_Eta", &towers_Eta);
  inputTree->SetBranchAddress("towers_Phi", &towers_Phi);

  // declare output tree
  // -------------------------------------------------------------------------//
  Int_t   out_jet_id;
  Int_t   out_jet_class;
  Float_t out_jet_Pt_og;    // for control
  Float_t out_jet_Mass_og;  // for control
  Float_t out_jet_Pt;
  Float_t out_jet_Eta;
  Float_t out_jet_Phi;
  Float_t out_jet_Mass;
  Float_t out_towers_Pt[230];
  Float_t out_towers_Eta[230];
  Float_t out_towers_Phi[230];
  Float_t out_nsubs_b05[45];
  Float_t out_nsubs_b10[45];
  Float_t out_nsubs_b20[45];

  outputTree->Branch("jet_id", &out_jet_id, "jet_id/I");
  outputTree->Branch("jet_class", &out_jet_class, "jet_class/I");
  outputTree->Branch("jet_Pt_og", &out_jet_Pt_og, "jet_Pt_og/F");
  outputTree->Branch("jet_Mass_og", &out_jet_Mass_og, "jet_Mass_og/F");
  outputTree->Branch("jet_Pt", &out_jet_Pt, "jet_Pt/F");
  outputTree->Branch("jet_Eta", &out_jet_Eta, "jet_Eta/F");
  outputTree->Branch("jet_Phi", &out_jet_Phi, "jet_Phi/F");
  outputTree->Branch("jet_Mass", &out_jet_Mass, "jet_Mass/F");
  outputTree->Branch("towers_Pt", &out_towers_Pt, "towers_Pt[230]/F");
  outputTree->Branch("towers_Eta", &out_towers_Eta, "towers_Eta[230]/F");
  outputTree->Branch("towers_Phi", &out_towers_Phi, "towers_Phi[230]/F");
  // had towers
  outputTree->Branch("nsubs_b05", &out_nsubs_b05, "nsubs_b05[45]/F");
  outputTree->Branch("nsubs_b10", &out_nsubs_b10, "nsubs_b10[45]/F");
  outputTree->Branch("nsubs_b20", &out_nsubs_b20, "nsubs_b20[45]/F");

  Int_t nentries = inputTree->GetEntries();
  cout << "Tree with " << nentries << " entries." << endl;
  for (Int_t i=0; i<nentries; i++) {
    inputTree->GetEntry(i);
    vector<PseudoJet> towers;
    for(Int_t j=0; j<230; j++) {
      if (towers_Pt[j]>0) {
        PseudoJet p(0.0, 0.0, 0.0, 0.0);
        p.reset_momentum_PtYPhiM(towers_Pt[j], towers_Eta[j], towers_Phi[j], 0.0);
        towers.push_back(p);
      }
      out_towers_Pt[j]  = towers_Pt[j];
      out_towers_Eta[j] = towers_Eta[j];
      out_towers_Phi[j] = towers_Phi[j];
    }

    ClusterSequence ReclusterConsts(towers, *jet_def);
    vector<PseudoJet> InclusiveJets = sorted_by_pt(ReclusterConsts.inclusive_jets());

    // if (InclusiveJets.size()<1)
    // {
    //   rejected_jets++;
    //   continue;
    // }

    PseudoJet leading_jet;
    leading_jet = InclusiveJets[0];

    out_jet_id = jet_id[0];
    out_jet_class = jet_class[0];
    out_jet_Pt_og = jet_Pt[0];
    out_jet_Mass_og = jet_Mass[0];
    out_jet_Pt = leading_jet.perp();
    out_jet_Mass = leading_jet.m();
    out_jet_Eta = leading_jet.eta();
    out_jet_Phi = leading_jet.phi_std();

    Int_t nsub_count = 0;
    for (Int_t k=1; k<46; k++) {
      out_nsubs_b05[nsub_count] = Nsubjettiness_Nbeta(leading_jet, k, 0.5);
      out_nsubs_b10[nsub_count] = Nsubjettiness_Nbeta(leading_jet, k, 1.0);
      out_nsubs_b20[nsub_count] = Nsubjettiness_Nbeta(leading_jet, k, 2.0);
      nsub_count++;
    }

    outputTree->Fill();
    if (i % 10000 == 0)
      cout << i << " jets " << endl;
  } // entry loop
}

void calculate_nsubs(const char *inputFile, const char *outputFile)
{

  TFile *inFile = new TFile(inputFile, "read");  //input file
  TTree *inputTree = (TTree*)inFile->Get("constituents");

  TFile *outFile = new TFile(outputFile, "recreate"); // output file
  TTree *outputTree = new TTree("HL", "nsubjettiness");

  analyzeTree(inputTree, outputTree);
  outFile->Write();
  outFile->Close();
  inFile->Close();

  //cout <<"done :)"<< endl;
}
