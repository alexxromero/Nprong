// -------------------------------------------------
// structure of the tree containing the
// centered jet constituents
// -------------------------------------------------

#include <stdio.h>
#include <iostream>

class JetBranch : public TObject
{
public:
  Int_t   jet_id[1], jet_class[1];
  Float_t jet_Pt[1], jet_Mass[1];
  Float_t towers_Pt[230], towers_Eta[230], towers_Phi[230];

  ClassDef(JetBranch, 1)
};
