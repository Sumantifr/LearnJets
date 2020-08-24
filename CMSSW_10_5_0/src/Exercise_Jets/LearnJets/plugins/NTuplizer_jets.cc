// -*- C++ -*-
//
// Package:    Exercise_Jets/LearnJets
// Class:   ExJets   
// 
//
// Original Author:  Suman Chatterjee
//         Created:  Sat, 22 Aug 2020 10:14:16 GMT
//
//

// system include files-
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TAxis.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "TRandom.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include "GeneratorInterface/Pythia8Interface/plugins/ReweightUserHooks.h"

//#include "fastjet"
//#include "fastjet/contrib/"

#include <string>

#include <iostream>
#include <fstream>

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include  "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
//#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Spring16GeneralPurpose.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReport.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "JetMETCorrections/Modules/interface/JetResolution.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
//#include "CondFormats/DataRecord/interface/JetResolutionRcd.h"
//#include "CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"

#include "SimDataFormats/GeneratorProducts/interface/PdfInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include <fastjet/GhostedAreaSpec.hh>
#include "fastjet/GhostedAreaSpec.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"
#include "fastjet/tools/MassDropTagger.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"
#include "fastjet/tools/GridMedianBackgroundEstimator.hh"
#include "fastjet/tools/Subtractor.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/Subtractor.hh"
#include "fastjet/contrib/SoftDrop.hh"
//

using namespace std;
using namespace edm;
using namespace reco;  
using namespace CLHEP;
using namespace trigger;
using namespace math;
using namespace fastjet;
using namespace fastjet::contrib;

const float mu_mass = 0.105658;
const float el_mass = 0.000511;
const float pival = acos(-1.);

double width = 1.2;

struct triggervar{
  HepLorentzVector trg4v;
  bool		       both;
  bool            level1;
  bool            highl;
  int             ihlt;
  int             prescl;
};

int getbinid(double val, int nbmx, double* array) {
  if (val<array[0]) return -2;
  for (int ix=0; ix<=nbmx; ix++) {
    if (val < array[ix]) return ix-1;
  }
  return -3;
}

double theta_to_eta(double theta) { return -log(tan(theta/2.)); }

double PhiInRange(const double& phi) {
  double phiout = phi;

  if( phiout > 2*M_PI || phiout < -2*M_PI) {
    phiout = fmod( phiout, 2*M_PI);
  }
  if (phiout <= -M_PI) phiout += 2*M_PI;
  else if (phiout >  M_PI) phiout -= 2*M_PI;

  return phiout;
}

double delta2R(double eta1, double phi1, double eta2, double phi2) {
  return sqrt(pow(eta1 - eta2,2) +pow(PhiInRange(phi1 - phi2),2));
}

double diff_func(double f1, double f2){
double ff = pow(f1-f2,2)*1./pow(f1+f2,2);
return ff;
}


TLorentzVector productX(TLorentzVector X, TLorentzVector Y, float pro1, float pro2)
{
float b1, b2, b3;
float c1, c2, c3;

b1 = X.Px();
b2 = X.Py();
b3 = X.Pz();

c1 = Y.Px();
c2 = Y.Py();
c3 = Y.Pz();

float d1, d2, e1, e2, X1, X2;

X1 = pro1;
X2 = pro2;

d1 = (c2*X1 - b2*X2)*1./(b1*c2 - b2*c1);
d2 = (c1*X1 - b1*X2)*1./(b2*c1 - b1*c2);
e1 = (b2*c3 - b3*c2)*1./(b1*c2 - b2*c1);
e2 = (b1*c3 - b3*c1)*1./(b2*c1 - b1*c2);

float A, B, C;
A = (e1*e1 + e2*e2+ 1);
B = 2*(d1*e1 + d2*e2);
C = d1*d1 + d2*d2 - 1;

float sol;

if((pow(B,2) - (4*A*C)) < 0){
sol = -1*B/(2*A);

float A1, A2, A3;
A3 = sol;
A1 = d1 + e1*A3;
A2 = d2 + e2*A3;

TLorentzVector vec4;
vec4.SetPxPyPzE(A1,A2,A3,0);
return vec4;
}
else{
float sol1 = (-1*B+sqrt((pow(B,2) - (4*A*C))))*1./(2*A);
float sol2 =  (-1*B-sqrt((pow(B,2) - (4*A*C))))*1./(2*A);
(sol1>sol2)?sol=sol1:sol=sol2;

float A1, A2, A3;
A3 = sol;
A1 = d1 + e1*A3;
A2 = d2 + e2*A3;

TLorentzVector vec4;
vec4.SetPxPyPzE(A1,A2,A3,0);
return vec4;;
}

}

//class declaration
//
class ExJets : public edm::EDAnalyzer {
   public:
      explicit ExJets(const edm::ParameterSet&);
      ~ExJets();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  void fillmetarray();
      // ----------member data ---------------------------
  int Nevt;
  int ncnt;
  bool isData;
  bool isMC;
  bool isSoftDrop;
  bool isReconstruct ;
  bool isHistFill;
  std::string theRootFileName;
  std::string theHLTTag;
  std::string softdropmass;
  
  int iTag;
  int iTagMET;
  double jtptthr;
  double minPt;
  double maxEta;
  double maxgenEta;
  double AK8PtCut;

  double JetRadius;
  double beta, beta1, beta2 ;
  double z_cut;
  double Rsub_trim, fcut_trim, Rcut_prun, zcut_prun;
  
  const float t_mass = 173;
  const float w_mass = 80.4;
  float b_mass;
  float mass_l = 0;
  
//  edm::EDGetTokenT<reco::BeamSpot> tok_beamspot_;
  edm::EDGetTokenT<double> tok_Rho_;
  edm::EDGetTokenT<reco::BeamSpot> tok_beamspot_;
  edm::EDGetTokenT<reco::VertexCollection> tok_primaryVertices_;
  //edm::EDGetTokenT<reco::GenParticleCollection>tok_genparticles_;
  edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection> tok_sv;
  edm::EDGetTokenT<pat::METCollection>tok_mets_ ;
  edm::EDGetTokenT<reco::GenMETCollection>tok_genmets_;
  
  edm::EDGetTokenT<edm::View<pat::Jet>>tok_pfjetAK8s_;
  edm::EDGetTokenT<edm::View<pat::Jet>>tok_pfjetAK8s_CHS_;
  edm::EDGetTokenT<edm::View<pat::Jet>>tok_pfjetAK8s_Puppi_;
  edm::EDGetTokenT<reco::GenJetCollection>tok_genjetAK8s_;
  edm::EDGetTokenT<edm::View<pat::Jet>>tok_pfjetAK4s_;
  edm::EDGetTokenT<reco::GenJetCollection>tok_genjetAK4s_;
//  edm::EDGetTokenT<edm::View<pat::PackedGenParticle>>tok_genparticles_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>>tok_genparticles_;
  
  edm::EDGetTokenT<HepMCProduct> tok_HepMC ;
  edm::EDGetTokenT<GenEventInfoProduct> tok_wt_;
  
  edm::EDGetTokenT<LHEEventProduct> lheEventProductToken_;
  edm::EDGetTokenT<GenEventInfoProduct> GenEventProductToken_;

  edm::EDGetTokenT<edm::View<pat::Muon>> tok_muons_;
  edm::EDGetTokenT<edm::View<pat::Electron>> tok_electrons_;
  edm::EDGetTokenT<edm::View<pat::Photon>>tok_photons_;

  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileup_;

  //edm::InputTag tag_mvaEleIDSpring16GeneralPurposeV1wp90_reco;
  //edm::InputTag tag_mvaEleIDSpring16GeneralPurposeV1wp90_noIso_reco;
  //edm::EDGetTokenT <edm::ValueMap <bool> > tok_mvaEleIDSpring16GeneralPurposeV1wp90_reco;
  //edm::EDGetTokenT <edm::ValueMap <bool> > tok_mvaEleIDSpring16GeneralPurposeV1wp90_noIso_reco; 
 
  edm::InputTag tag_mvaPhoIDSpring16GeneralPurposeV1wp90_reco;
  edm::EDGetTokenT <edm::ValueMap <bool> > tok_mvaPhoIDSpring16GeneralPurposeV1wp90_reco;

  edm::EDGetTokenT<edm::TriggerResults> triggerBits_;
  edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjects_;
  edm::EDGetTokenT<pat::PackedTriggerPrescales> triggerPrescales_;

  TFile* theFile;

  TTree* T1;
  
 // HLTConfigProvider hltConfig_;
  
  unsigned ievt;
  
  static const int njetmx = 20; 
  static const int njetmxAK8 =10;
  static const int npartmx = 50; 
  
  int irunold;
  int irun, ilumi, ifltr, nprim, ibrnch;
  double event_weight;
  double weights[njetmx];
  
  double Rho ;

  int npfjetAK8;
  float pfjetAK8pt[njetmxAK8], pfjetAK8y[njetmxAK8], pfjetAK8eta[njetmxAK8], pfjetAK8phi[njetmxAK8], pfjetAK8mass[njetmxAK8];
  float pfjetAK8sdmass[njetmxAK8], pfjetAK8sdmass1[njetmxAK8], pfjetAK8sdmass2[njetmxAK8], pfjetAK8trimmass[njetmxAK8], pfjetAK8prunmass[njetmxAK8];
  bool  pfjetAK8tightID[njetmxAK8];
  float pfjetAK8JEC[njetmxAK8], pfjetAK8JECL1[njetmxAK8], pfjetAK8JECL2[njetmxAK8], pfjetAK8JECL3[njetmxAK8], pfjetAK8JECL2L3[njetmxAK8];
  float pfjetAK8reso[njetmxAK8];
  
  int npfjetAK8_CHS;
  float pfjetAK8pt_CHS[njetmxAK8], pfjetAK8y_CHS[njetmxAK8], pfjetAK8eta_CHS[njetmxAK8], pfjetAK8phi_CHS[njetmxAK8], pfjetAK8mass_CHS[njetmxAK8];
  float pfjetAK8sdmass_CHS[njetmxAK8], pfjetAK8sdmass1_CHS[njetmxAK8], pfjetAK8sdmass2_CHS[njetmxAK8], pfjetAK8trimmass_CHS[njetmxAK8], pfjetAK8prunmass_CHS[njetmxAK8];
  bool  pfjetAK8tightID_CHS[njetmxAK8];
  float pfjetAK8reso_CHS[njetmxAK8];
  
  int npfjetAK8_Puppi;
  float pfjetAK8pt_Puppi[njetmxAK8], pfjetAK8y_Puppi[njetmxAK8], pfjetAK8eta_Puppi[njetmxAK8], pfjetAK8phi_Puppi[njetmxAK8], pfjetAK8mass_Puppi[njetmxAK8];
  float pfjetAK8sdmass_Puppi[njetmxAK8], pfjetAK8sdmass1_Puppi[njetmxAK8], pfjetAK8sdmass2_Puppi[njetmxAK8], pfjetAK8trimmass_Puppi[njetmxAK8], pfjetAK8prunmass_Puppi[njetmxAK8];
  bool  pfjetAK8tightID_Puppi[njetmxAK8];
  float pfjetAK8reso_Puppi[njetmxAK8];
  
  int npfjetAK4;
  float pfjetAK4pt[njetmxAK8], pfjetAK4y[njetmxAK8], pfjetAK4eta[njetmxAK8], pfjetAK4phi[njetmxAK8], pfjetAK4mass[njetmxAK8];
  float pfjetAK4sdmass[njetmxAK8], pfjetAK4filmass[njetmxAK8], pfjetAK4trimmass[njetmxAK8], pfjetAK4prunmass[njetmxAK8];
  bool  pfjetAK4tightID[njetmxAK8];
  float pfjetAK4JEC[njetmxAK8], pfjetAK4JECL1[njetmxAK8], pfjetAK4JECL2[njetmxAK8], pfjetAK4JECL3[njetmxAK8], pfjetAK4JECL2L3[njetmxAK8];
  float pfjetAK4reso[njetmxAK8];
  
  int ngenjetAK8;
  float genjetAK8pt[njetmx], genjetAK8y[njetmx], genjetAK8phi[njetmx], genjetAK8mass[njetmx];
  float genjetAK8sdmass[njetmx], genjetAK8sdmass1[njetmx], genjetAK8sdmass2[njetmx], genjetAK8trimmass[njetmx], genjetAK8prunmass[njetmx];
 
  int ngenjetAK4;
  float genjetAK4pt[njetmx], genjetAK4y[njetmx], genjetAK4phi[njetmx], genjetAK4btag[njetmx], genjetAK4mass[njetmx]; 
 
  int ngenparticles;
  int genpartstatus[npartmx], genpartpdg[npartmx], genpartmompdg[npartmx], genpartmomid[npartmx], genpartdaugno[npartmx];
  float genpartpt[npartmx], genparteta[npartmx], genpartphi[npartmx], genpartm[npartmx], genpartq[npartmx];
  bool genpartfromhard[npartmx], genpartfromhardbFSR[npartmx], genpartisPromptFinalState[npartmx], genpartisLastCopyBeforeFSR[npartmx];
  
  int ngenjetantikt;
  float genjetantiktpt[njetmx], genjetantikty[njetmx], genjetantiktphi[njetmx], genjetantiktmass[njetmx]; 
 
  int ngenjetkt;
  float genjetktpt[njetmx], genjetkty[njetmx], genjetktphi[njetmx], genjetktmass[njetmx]; 
 
  int ngenjetca;
  float genjetcapt[njetmx], genjetcay[njetmx], genjetcaphi[njetmx], genjetcamass[njetmx]; 
 
 
  float miset , misphi , sumEt;
  float genmiset, genmisphi;

  int nmuons;
  float muonp[njetmx], muone[njetmx], muonpt[njetmx], muoneta[njetmx], muonphi[njetmx], muondrbm[njetmx], muondz[njetmx], muonpter[njetmx], muonchi[njetmx], muonecal[njetmx], muonhcal[njetmx], muonemiso[njetmx], muonhadiso[njetmx], muontkpt03[njetmx], muontkpt05[njetmx];
  float muonposmatch[njetmx], muontrkink[njetmx], muonsegcom[njetmx], muonpfiso[njetmx], muontrkvtx[njetmx], muonhit[njetmx], muonpixhit[njetmx], muonmst[njetmx], muontrklay[njetmx], muonvalfrac[njetmx];
  int muonndf[njetmx];
  bool muonisPF[njetmx], muonisGL[njetmx], muonisTRK[njetmx];
  bool muonisGoodGL[njetmx], muonisMed[njetmx], muonisLoose[njetmx];

  int nelecs;
  bool elmvaid[njetmx], elmvaid_noIso[njetmx];
  float elpt[njetmx], eleta[njetmx], elphi[njetmx], ele[njetmx], elp[njetmx], eldxy[njetmx],  eldxy_sv[njetmx], eldz[njetmx], elhovere[njetmx], elqovrper[njetmx], elchi[njetmx], elemiso03[njetmx], elhadiso03[njetmx], elemiso04[njetmx], elhadiso04[njetmx], elhadisodepth03[njetmx], eltkpt03[njetmx], eltkpt04[njetmx], eleoverp[njetmx], elietaieta[njetmx], eletain[njetmx], elphiin[njetmx], elfbrem[njetmx], elchhadiso03[njetmx], elchhadiso04[njetmx], elnohits[njetmx], elmisshits[njetmx] ;
  float elchhadiso[njetmx], elneuhadiso[njetmx], elphoiso[njetmx], elpuchhadiso[njetmx], elpfiso[njetmx], elconvdist[njetmx], elconvdoct[njetmx];
  int elndf[njetmx];

  int nphotons;
  bool phomvaid[njetmx];
  float phoe[njetmx], phoeta[njetmx], phophi[njetmx], phoe1by9[njetmx], phoe9by25[njetmx], phohadbyem[njetmx], photrkiso[njetmx], phoemiso[njetmx], phohadiso[njetmx], phochhadiso[njetmx], phoneuhadiso[njetmx], phoPUiso[njetmx], phophoiso[njetmx], phoietaieta[njetmx];
  
  int ntrigobjs;
  float trigobjpt[njetmx], trigobjeta[njetmx],trigobjphi[njetmx], trigobje[njetmx];
  bool trigobjHLT[njetmx], trigobjL1[njetmx],  trigobjBoth[njetmx];
  int  trigobjIhlt[njetmx];
  
  unsigned int mypow_2[32];

  float qscale;
  float wtfact , weight2 = 1.0;
  int npu_vert;
  
  int nchict;
  int nvert;;
  int ndofct;
  
  static const int nHLTmx = 3;
  const char *hlt_name[nHLTmx] = {"HLT_IsoMu24_v","HLT_Ele32_WPTight_Gsf_v","HLT_AK8PFJet500_v"};
  int ihlt01, ihlt02, ihlt03;
  float prescl01, prescl02, prescl03;
  double compres[nHLTmx] = {0};
  int trig_value;
  float weighttrg[nHLTmx];
  
  HLTPrescaleProvider hltPrescaleProvider_;
  
  // ---- Jet Corrector Parameter End---- //
  
  // ---- Jet Corrector Parameter ---- //
  JetCorrectorParameters *L1FastAK4, *L2RelativeAK4, *L3AbsoluteAK4, *L2L3ResidualAK4;
  vector<JetCorrectorParameters> vecL1FastAK4, vecL2RelativeAK4, vecL3AbsoluteAK4, vecL2L3ResidualAK4;
  FactorizedJetCorrector *jecL1FastAK4, *jecL2RelativeAK4, *jecL3AbsoluteAK4, *jecL2L3ResidualAK4;
 
  JetCorrectorParameters *L1FastAK8, *L2RelativeAK8, *L3AbsoluteAK8, *L2L3ResidualAK8;
  vector<JetCorrectorParameters> vecL1FastAK8, vecL2RelativeAK8, vecL3AbsoluteAK8, vecL2L3ResidualAK8;
  FactorizedJetCorrector *jecL1FastAK8, *jecL2RelativeAK8, *jecL3AbsoluteAK8, *jecL2L3ResidualAK8;
 

  // std::string mFileName,mPuFileName,mPuTrigName;
  std::string mJECL1FastFileAK4, mJECL2RelativeFileAK4, mJECL3AbsoluteFileAK4, mJECL2L3ResidualFileAK4, mJECL1FastFileAK8, mJECL2RelativeFileAK8, mJECL3AbsoluteFileAK8, mJECL2L3ResidualFileAK8;
  std::string mPtResoFileAK4, mPtResoFileAK8, mPtSFFileAK4, mPtSFFileAK8, mPtResoFileAK8CHS, mPtResoFileAK8Puppi, mPtSFFileAK8CHS, mPtSFFileAK8Puppi;
  // ---- Jet Corrector Parameter End---- //
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

ExJets::ExJets(const edm::ParameterSet& pset):
hltPrescaleProvider_(pset, consumesCollector(), *this)
{
   //now do what ever initialization is needed
  
  edm::Service<TFileService> fs;
  
  isData    = pset.getUntrackedParameter<bool>("Data",false);
  isMC      = pset.getUntrackedParameter<bool>("MonteCarlo", false);
  isSoftDrop      = pset.getUntrackedParameter<bool>("SoftDrop_ON",false);
  theRootFileName = pset.getUntrackedParameter<string>("RootFileName");
  theHLTTag = pset.getUntrackedParameter<string>("HLTTag", "HLT");
  minPt = pset.getUntrackedParameter<double>("minPt",100.);
  maxEta = pset.getUntrackedParameter<double>("maxEta",3.);
  maxgenEta = pset.getUntrackedParameter<double>("maxgenEta",3.);
  AK8PtCut = pset.getUntrackedParameter<double>("AK8PtCut",250.);
 
  triggerBits_ = consumes<edm::TriggerResults> ( pset.getParameter<edm::InputTag>("bits"));
  triggerObjects_ = consumes<pat::TriggerObjectStandAloneCollection>(pset.getParameter<edm::InputTag>("objects"));
  triggerPrescales_ = consumes<pat::PackedTriggerPrescales>(pset.getParameter<edm::InputTag>("prescales"));
  
  softdropmass = pset.getUntrackedParameter<string>("softdropmass");
  
  tok_beamspot_ = consumes<reco::BeamSpot> (pset.getParameter<edm::InputTag>("Beamspot"));
  tok_primaryVertices_ =consumes<reco::VertexCollection>( pset.getParameter<edm::InputTag>("PrimaryVertices"));
  //slimmedSecondaryVertices
  tok_sv =consumes<reco::VertexCompositePtrCandidateCollection>( pset.getParameter<edm::InputTag>("SecondaryVertices"));
  
  tok_Rho_ = consumes<double>(pset.getParameter<edm::InputTag>("PFRho"));
   
  tok_mets_= consumes<pat::METCollection> ( pset.getParameter<edm::InputTag>("PFMet"));
  tok_genmets_= consumes<reco::GenMETCollection> ( pset.getParameter<edm::InputTag>("GENMet"));
 
  tok_muons_ = consumes<edm::View<pat::Muon>> ( pset.getParameter<edm::InputTag>("Muons"));
  tok_electrons_ = consumes<edm::View<pat::Electron>> ( pset.getParameter<edm::InputTag>("Electrons"));
  tok_photons_ = consumes<edm::View<pat::Photon>>  ( pset.getParameter<edm::InputTag>("Photons"));
 
  tok_pfjetAK8s_= consumes<edm::View<pat::Jet>>( pset.getParameter<edm::InputTag>("PFJetsAK8"));
  tok_pfjetAK8s_CHS_= consumes<edm::View<pat::Jet>>( pset.getParameter<edm::InputTag>("PFCHSJetsAK8"));
  tok_pfjetAK8s_Puppi_= consumes<edm::View<pat::Jet>>( pset.getParameter<edm::InputTag>("PFPuppiJetsAK8"));
  tok_pfjetAK4s_= consumes<edm::View<pat::Jet>>( pset.getParameter<edm::InputTag>("PFJetsAK4"));
   if(isMC){
      tok_genjetAK8s_= consumes<reco::GenJetCollection>( pset.getParameter<edm::InputTag>("GENJetAK8"));
      tok_genjetAK4s_= consumes<reco::GenJetCollection>( pset.getParameter<edm::InputTag>("GENJetAK4"));
//	  tok_genparticles_ = consumes<edm::View<pat::PackedGenParticle>>( pset.getParameter<edm::InputTag>("GenParticles"));
	  tok_genparticles_ = consumes<std::vector<reco::GenParticle>>( pset.getParameter<edm::InputTag>("GenParticles"));
   }
 
  tag_mvaPhoIDSpring16GeneralPurposeV1wp90_reco = pset.getParameter<edm::InputTag>("label_mvaPhoIDSpring16GeneralPurposeV1wploose_reco");
  tok_mvaPhoIDSpring16GeneralPurposeV1wp90_reco = consumes<edm::ValueMap <bool> >(tag_mvaPhoIDSpring16GeneralPurposeV1wp90_reco);

  mJECL1FastFileAK4         = pset.getParameter<std::string>("jecL1FastFileAK4");
  mJECL1FastFileAK8         = pset.getParameter<std::string>("jecL1FastFileAK8");
  mJECL2RelativeFileAK4     = pset.getParameter<std::string>("jecL2RelativeFileAK4");
  mJECL2RelativeFileAK8     = pset.getParameter<std::string>("jecL2RelativeFileAK8");
  mJECL3AbsoluteFileAK4     = pset.getParameter<std::string>("jecL3AbsoluteFileAK4");
  mJECL3AbsoluteFileAK8     = pset.getParameter<std::string> ("jecL3AbsoluteFileAK8");
  mJECL2L3ResidualFileAK4   = pset.getParameter<std::string> ("jecL2L3ResidualFileAK4");
  mJECL2L3ResidualFileAK8   = pset.getParameter<std::string> ("jecL2L3ResidualFileAK8");
  
  mPtResoFileAK4  = pset.getParameter<std::string>("PtResoFileAK4");
  mPtResoFileAK8  = pset.getParameter<std::string>("PtResoFileAK8");
  mPtResoFileAK8CHS  = pset.getParameter<std::string>("PtResoFileAK8CHS");
  mPtResoFileAK8Puppi  = pset.getParameter<std::string>("PtResoFileAK8Puppi");
  mPtSFFileAK4  = pset.getParameter<std::string>("PtSFFileAK4");
  mPtSFFileAK8  = pset.getParameter<std::string>("PtSFFileAK8");
  mPtSFFileAK8CHS  = pset.getParameter<std::string>("PtSFFileAK8CHS");
  mPtSFFileAK8Puppi  = pset.getParameter<std::string>("PtSFFileAK8Puppi");
  
  if(isMC){    
    tok_HepMC = consumes<HepMCProduct>(pset.getParameter<edm::InputTag>("Generator"));
    tok_wt_ = consumes<GenEventInfoProduct>(pset.getParameter<edm::InputTag>("Generator")) ;
    pileup_ = consumes<std::vector<PileupSummaryInfo> >(pset.getParameter<edm::InputTag>("slimmedAddPileupInfo"));
	} 
 
  beta = pset.getUntrackedParameter<double>("beta",0.);
  beta1 = pset.getUntrackedParameter<double>("beta1",0.5);
  beta2 = pset.getUntrackedParameter<double>("beta2",1.0);
  z_cut = pset.getUntrackedParameter<double>("z_cut",0.1); 
  Rsub_trim = pset.getUntrackedParameter<double>("Rsub_trim",0.2);
  fcut_trim = pset.getUntrackedParameter<double>("fcut_trim",0.3);
  Rcut_prun = pset.getUntrackedParameter<double>("Rcut_prun",0.2);
  zcut_prun = pset.getUntrackedParameter<double>("zcut_prun",0.1);
 
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();
 
  //  T1 = fs->make<TTree>("T1","PFandgenjetAK8AK8s") ;
  T1 = new TTree("T1", "WPrimeNtuple");

  T1->Branch("irun", &irun, "irun/I");  
  T1->Branch("ilumi", &ilumi, "ilumi/I");  
  
  T1->Branch("ievt", &ievt, "ievt/i");
  T1->Branch("nprim", &nprim, "nprim/I");
  T1->Branch("nvert", &nvert, "nvert/I");  
  T1->Branch("ndofct", &ndofct, "ndofct/I");
  T1->Branch("nchict", &nchict, "nchict/I");

  T1->Branch("Rho", &Rho, "Rho/D") ;
  
  T1->Branch("event_weight", &event_weight, "event_weight/D") ;
  T1->Branch("qscale",&qscale,"qscale/F");
  T1->Branch("npu_vert",&npu_vert,"npu_vert/I");

  T1->Branch("trig_value",&trig_value,"trig_value/I");  
  T1->Branch("ihlt01",&ihlt01,"ihlt01/I");
  T1->Branch("ihlt02",&ihlt02,"ihlt02/I");
  T1->Branch("ihlt03",&ihlt03,"ihlt03/I");
  T1->Branch("prescl01",&prescl01,"prescl01/F");
  T1->Branch("prescl02",&prescl02,"prescl02/F");
  T1->Branch("prescl03",&prescl03,"prescl03/F");
 
  
  T1->Branch("ntrigobjs",&ntrigobjs,"ntrigobjs/I");
  T1->Branch("trigobjpt",trigobjpt,"trigobjpt[ntrigobjs]/F");
  T1->Branch("trigobjeta",trigobjeta,"trigobjeta[ntrigobjs]/F");
  T1->Branch("trigobjphi",trigobjphi,"trigobjphi[ntrigobjs]/F");
  T1->Branch("trigobje",trigobje,"trigobje[ntrigobjs]/F");
  T1->Branch("trigobjHLT",trigobjHLT,"trigobjHLT[ntrigobjs]/O");
  T1->Branch("trigobjL1",trigobjL1,"trigobjL1[ntrigobjs]/O");
  T1->Branch("trigobjBoth",trigobjBoth,"trigobjBoth[ntrigobjs]/O");
  T1->Branch("trigobjIhlt",trigobjIhlt,"trigobjIhlt[ntrigobjs]/I");
  
  T1->Branch("PFMET",&miset,"miset/F") ;
  T1->Branch("PFMETPhi",&misphi,"misphi/F") ;
  T1->Branch("sumEt",&sumEt,"sumEt/F");

  T1->Branch("npfjetAK8",&npfjetAK8, "npfjetAK8/I"); 
  T1->Branch("pfjetAK8tightID",&pfjetAK8tightID,"pfjetAK8tightID[npfjetAK8]/O");
  T1->Branch("pfjetAK8pt",pfjetAK8pt,"pfjetAK8pt[npfjetAK8]/F");
  T1->Branch("pfjetAK8y",pfjetAK8y,"pfjetAK8y[npfjetAK8]/F");
  T1->Branch("pfjetAK8eta",pfjetAK8eta,"pfjetAK8eta[npfjetAK8]/F");
  T1->Branch("pfjetAK8phi",pfjetAK8phi,"pfjetAK8phi[npfjetAK8]/F");
  T1->Branch("pfjetAK8mass",pfjetAK8mass,"pfjetAK8mass[npfjetAK8]/F");
  T1->Branch("pfjetAK8JEC",pfjetAK8JEC,"pfjetAK8JEC[npfjetAK8]/F");
  T1->Branch("pfjetAK8JECL1",pfjetAK8JECL1,"pfjetAK8JECL1[npfjetAK8]/F");
  T1->Branch("pfjetAK8JECL2",pfjetAK8JECL2,"pfjetAK8JECL2[npfjetAK8]/F");
  T1->Branch("pfjetAK8JECL3",pfjetAK8JECL3,"pfjetAK8JECL3[npfjetAK8]/F");
  T1->Branch("pfjetAK8JECL2L3",pfjetAK8JECL2L3,"pfjetAK8JECL2L3[npfjetAK8]/F"); 
  T1->Branch("pfjetAK8reso",pfjetAK8reso,"pfjetAK8reso[npfjetAK8]/F");
  T1->Branch("pfjetAK8sdmass",pfjetAK8sdmass,"pfjetAK8sdmass[npfjetAK8]/F");
  T1->Branch("pfjetAK8sdmass1",pfjetAK8sdmass1,"pfjetAK8sdmass1[npfjetAK8]/F");
  T1->Branch("pfjetAK8sdmass2",pfjetAK8sdmass2,"pfjetAK8sdmass2[npfjetAK8]/F");
  T1->Branch("pfjetAK8trimmass",pfjetAK8trimmass,"pfjetAK8trimmass[npfjetAK8]/F");
  T1->Branch("pfjetAK8prunmass",pfjetAK8prunmass,"pfjetAK8prunmass[npfjetAK8]/F");
  
  T1->Branch("npfjetAK8_CHS",&npfjetAK8_CHS, "npfjetAK8_CHS/I"); 
  T1->Branch("pfjetAK8tightID_CHS",&pfjetAK8tightID_CHS,"pfjetAK8tightID_CHS[npfjetAK8_CHS]/O");
  T1->Branch("pfjetAK8pt_CHS",pfjetAK8pt_CHS,"pfjetAK8pt_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8y_CHS",pfjetAK8y_CHS,"pfjetAK8y_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8eta_CHS",pfjetAK8eta_CHS,"pfjetAK8eta_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8phi_CHS",pfjetAK8phi_CHS,"pfjetAK8phi_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8mass_CHS",pfjetAK8mass_CHS,"pfjetAK8mass_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8reso_CHS",pfjetAK8reso_CHS,"pfjetAK8reso_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8sdmass_CHS",pfjetAK8sdmass_CHS,"pfjetAK8sdmass_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8sdmass1_CHS",pfjetAK8sdmass1_CHS,"pfjetAK8sdmass1_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8sdmass2_CHS",pfjetAK8sdmass2_CHS,"pfjetAK8sdmass2_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8trimmass_CHS",pfjetAK8trimmass_CHS,"pfjetAK8trimmass_CHS[npfjetAK8_CHS]/F");
  T1->Branch("pfjetAK8prunmass_CHS",pfjetAK8prunmass_CHS,"pfjetAK8prunmass_CHS[npfjetAK8_CHS]/F");
  
  T1->Branch("npfjetAK8_Puppi",&npfjetAK8_Puppi, "npfjetAK8_Puppi/I"); 
  T1->Branch("pfjetAK8tightID_Puppi",&pfjetAK8tightID_Puppi,"pfjetAK8tightID_Puppi[npfjetAK8_Puppi]/O");
  T1->Branch("pfjetAK8pt_Puppi",pfjetAK8pt_Puppi,"pfjetAK8pt_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8y_Puppi",pfjetAK8y_Puppi,"pfjetAK8y_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8eta_Puppi",pfjetAK8eta_Puppi,"pfjetAK8eta_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8phi_Puppi",pfjetAK8phi_Puppi,"pfjetAK8phi_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8mass_Puppi",pfjetAK8mass_Puppi,"pfjetAK8mass_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8reso_Puppi",pfjetAK8reso_Puppi,"pfjetAK8reso_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8sdmass_Puppi",pfjetAK8sdmass_Puppi,"pfjetAK8sdmass_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8sdmass1_Puppi",pfjetAK8sdmass1_Puppi,"pfjetAK8sdmass1_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8sdmass2_Puppi",pfjetAK8sdmass2_Puppi,"pfjetAK8sdmass2_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8trimmass_Puppi",pfjetAK8trimmass_Puppi,"pfjetAK8trimmass_Puppi[npfjetAK8_Puppi]/F");
  T1->Branch("pfjetAK8prunmass_Puppi",pfjetAK8prunmass_Puppi,"pfjetAK8prunmass_Puppi[npfjetAK8_CHS]/F");
  
  
  T1->Branch("npfjetAK4",&npfjetAK4,"npfjetAK4/I"); 
  T1->Branch("pfjetAK4tightID",&pfjetAK4tightID,"pfjetAK4tightID[npfjetAK4]/O");
  T1->Branch("pfjetAK4pt",pfjetAK4pt,"pfjetAK4pt[npfjetAK4]/F");
  T1->Branch("pfjetAK4eta",pfjetAK4eta,"pfjetAK4eta[npfjetAK4]/F");
  T1->Branch("pfjetAK4y",pfjetAK4y,"pfjetAK4y[npfjetAK4]/F");
  T1->Branch("pfjetAK4phi",pfjetAK4phi,"pfjetAK4phi[npfjetAK4]/F");
  T1->Branch("pfjetAK4mass",pfjetAK4mass,"pfjetAK4mass[npfjetAK4]/F");
  T1->Branch("pfjetAK4JEC",pfjetAK4JEC,"pfjetAK4JEC[npfjetAK4]/F");
  T1->Branch("pfjetAK4JECL1",pfjetAK4JECL1,"pfjetAK4JECL1[npfjetAK4]/F");
  T1->Branch("pfjetAK4JECL2",pfjetAK4JECL2,"pfjetAK4JECL2[npfjetAK4]/F");
  T1->Branch("pfjetAK4JECL3",pfjetAK4JECL3,"pfjetAK4JECL3[npfjetAK4]/F");
  T1->Branch("pfjetAK4JECL2L3",pfjetAK4JECL2L3,"pfjetAK4JECL2L3[npfjetAK4]/F");
  T1->Branch("pfjetAK4reso",pfjetAK4reso,"pfjetAK4reso[npfjetAK4]/F");
  

  if(isMC){
 
  T1->Branch("GENMET",&genmiset,"genmiset/F") ;
  T1->Branch("GENMETPhi",&genmisphi,"genmisphi/F") ;
 
  T1->Branch("ngenjetAK8",&ngenjetAK8, "ngenjetAK8/I");
  T1->Branch("genjetAK8pt",genjetAK8pt,"genjetAK8pt[ngenjetAK8]/F");
  T1->Branch("genjetAK8y",genjetAK8y,"genjetAK8y[ngenjetAK8]/F");
  T1->Branch("genjetAK8phi",genjetAK8phi,"genjetAK8phi[ngenjetAK8]/F");
  T1->Branch("genjetAK8mass",genjetAK8mass,"genjetAK8mass[ngenjetAK8]/F");
  T1->Branch("genjetAK8sdmass",genjetAK8sdmass,"genjetAK8sdmass[ngenjetAK8]/F");
  T1->Branch("genjetAK8sdmass1",genjetAK8sdmass2,"genjetAK8sdmass1[ngenjetAK8]/F");
  T1->Branch("genjetAK8sdmass1",genjetAK8sdmass2,"genjetAK8sdmass2[ngenjetAK8]/F");
  T1->Branch("genjetAK8trimmass",genjetAK8trimmass,"genjetAK8trimmass[ngenjetAK8]/F");
  T1->Branch("genjetAK8prunmass",genjetAK8prunmass,"genjetAK8prunmass[ngenjetAK8]/F");
 
  T1->Branch("ngenjetAK4",&ngenjetAK4, "ngenjetAK4/I");
  T1->Branch("genjetAK4pt",genjetAK4pt,"genjetAK4pt[ngenjetAK4]/F");
  T1->Branch("genjetAK4y",genjetAK4y,"genjetAK4y[ngenjetAK4]/F");
  T1->Branch("genjetAK4phi",genjetAK4phi,"genjetAK4phi[ngenjetAK4]/F");
  T1->Branch("genjetAK4mass",genjetAK4mass,"genjetAK4mass[ngenjetAK4]/F");
  
  T1->Branch("ngenparticles",&ngenparticles, "ngenparticles/I");
  T1->Branch("genpartstatus",genpartstatus,"genpartstatus[ngenparticles]/I");
  T1->Branch("genpartpdg",genpartpdg,"genpartpdg[ngenparticles]/I");
  T1->Branch("genpartmompdg",genpartmompdg,"genpartmompdg[ngenparticles]/I");
//  T1->Branch("genpartmomid",genpartmomid,"genpartmomid[ngenparticles]/I");
  T1->Branch("genpartdaugno",genpartdaugno,"genpartdaugno[ngenparticles]/I");
  T1->Branch("genpartfromhard",genpartfromhard,"genpartfromhard[ngenparticles]/O");
  T1->Branch("genpartfromhardbFSR",genpartfromhardbFSR,"genpartfromhardbFSR[ngenparticles]/O");
  T1->Branch("genpartisPromptFinalState",genpartisPromptFinalState,"genpartisPromptFinalState[ngenparticles]/O");
  T1->Branch("genpartisLastCopyBeforeFSR",genpartisLastCopyBeforeFSR,"genpartisLastCopyBeforeFSR[ngenparticles]/O");
  T1->Branch("genpartpt",genpartpt,"genpartpt[ngenparticles]/F");
  T1->Branch("genparteta",genparteta,"genparteta[ngenparticles]/F");
  T1->Branch("genpartphi",genpartphi,"genpartphi[ngenparticles]/F");
  T1->Branch("genpartm",genpartm,"genpartm[ngenparticles]/F");
  T1->Branch("genpartq",genpartq,"genpartq[ngenparticles]/F");
  
  T1->Branch("ngenjetantikt",&ngenjetantikt,"ngenjetantikt/I");
  T1->Branch("genjetantiktpt",genjetantiktpt,"genjetantiktpt[ngenjetantikt]/F");
  T1->Branch("genjetantikty",genjetantikty,"genjetantikty[ngenjetantikt]/F");
  T1->Branch("genjetantiktphi",genjetantiktphi,"genjetantiktphi[ngenjetantikt]/F");
  T1->Branch("genjetantiktmass",genjetantiktmass,"genjetantiktmass[ngenjetantikt]/F");
  
  T1->Branch("ngenjetkt",&ngenjetkt,"ngenjetkt/I");
  T1->Branch("genjetktpt",genjetktpt,"genjetktpt[ngenjetkt]/F");
  T1->Branch("genjetkty",genjetkty,"genjetkty[ngenjetkt]/F");
  T1->Branch("genjetktphi",genjetktphi,"genjetktphi[ngenjetkt]/F");
  T1->Branch("genjetktmass",genjetktmass,"genjetktmass[ngenjetkt]/F");
  
  T1->Branch("ngenjetca",&ngenjetca,"ngenjetca/I");
  T1->Branch("genjetcapt",genjetcapt,"genjetcapt[ngenjetca]/F");
  T1->Branch("genjetcay",genjetcay,"genjetcay[ngenjetca]/F");
  T1->Branch("genjetcaphi",genjetcaphi,"genjetcaphi[ngenjetca]/F");
  T1->Branch("genjetcamass",genjetcamass,"genjetcamass[ngenjetca]/F");
  
 
  } //isMC


  T1->Branch("nmuons",&nmuons,"nmuons/I");
  T1->Branch("muonisPF",muonisPF,"muonisPF[nmuons]/O");
  T1->Branch("muonisGL",muonisGL,"muonisPF[nmuons]/O");
  T1->Branch("muonisTRK",muonisTRK,"muonisTRK[nmuons]/O");
  T1->Branch("muonisLoose",muonisLoose,"muonisLoose[nmuons]/O");
  T1->Branch("muonisGoodGL",muonisGoodGL,"muonisGoodGL[nmuons]/O");
  T1->Branch("muonisMed",muonisMed,"muonisMed[nmuons]/O");
  T1->Branch("muonpt",muonpt,"muonpt[nmuons]/F");
  T1->Branch("muonp",muonp,"muonp[nmuons]/F");
  T1->Branch("muone",muone,"muone[nmuons]/F");
  T1->Branch("muoneta",muoneta,"muoneta[nmuons]/F");
  T1->Branch("muonphi",muonphi,"muonphi[nmuons]/F");
  T1->Branch("muondrbm",muondrbm,"muondrbm[nmuons]/F");
  T1->Branch("muontrkvtx",muontrkvtx,"muontrkvtx[nmuons]/F");
  T1->Branch("muondz",muondz,"muondz[nmuons]/F");
  T1->Branch("muonpter",muonpter,"muonpter[nmuons]/F");
  T1->Branch("muonchi",muonchi,"muonchi[nmuons]/F");
  T1->Branch("muonndf",muonndf,"muonndf[nmuons]/I");
  T1->Branch("muonecal",muonecal,"muonecal[nmuons]/F");
  T1->Branch("muonhcal",muonhcal,"muonhcal[nmuons]/F");
  T1->Branch("muonemiso",muonemiso,"muonemiso[nmuons]/F");
  T1->Branch("muonhadiso",muonhadiso,"muonhadiso[nmuons]/F");
  T1->Branch("muonpfiso",muonpfiso,"muonpfiso[nmuons]/F");
  T1->Branch("muontkpt03",muontkpt03,"muontkpt03[nmuons]/F");
  T1->Branch("muontkpt05",muontkpt05,"muontkpt05[nmuons]/F");
  T1->Branch("muonposmatch",muonposmatch,"muonposmatch[nmuons]/F");
  T1->Branch("muontrkink",muontrkink,"muontrkink[nmuons]/F");
  T1->Branch("muonsegcom",muonsegcom,"muonsegcom[nmuons]/F");
  T1->Branch("muonthit",muonhit,"muonhit[nmuons]/F");
  T1->Branch("muonpixhit",muonpixhit,"muonpixhit[nmuons]/F");
  T1->Branch("muonmst",muonmst,"muonmst[nmuons]/F");
  T1->Branch("muontrklay",muontrklay,"muontrklay[nmuons]/F"); 
  T1->Branch("muonvalfrac",muonvalfrac,"muonvalfrac[nmuons]/F"); 
 
 
  T1->Branch("nelecs",&nelecs,"nelecs/I");
  T1->Branch("elpt",elpt,"elpt[nelecs]/F");
  T1->Branch("eleta",eleta,"eleta[nelecs]/F");
  T1->Branch("elphi",elphi,"elphi[nelecs]/F");
  T1->Branch("elp",elp,"elp[nelecs]/F");
  T1->Branch("ele",ele,"ele[nelecs]/F");
  T1->Branch("elmvaid",elmvaid,"elmvaid[nelecs]/O");
  T1->Branch("elmvaid_noIso",elmvaid_noIso,"elmvaid_noIso[nelecs]/O");
  T1->Branch("eldxy",eldxy,"eldxy[nelecs]/F");
  T1->Branch("eldxy_sv",eldxy_sv,"eldxy_sv[nelecs]/F");
  T1->Branch("eldz",eldz,"eldz[nelecs]/F");
  T1->Branch("elhovere",elhovere,"elhovere[nelecs]/F");
  
  
  T1->Branch("nphotons",&nphotons,"nphotons/I");
  T1->Branch("phoe",phoe,"phoe[nphotons]/F");
  T1->Branch("phoeta",phoeta,"phoeta[nphotons]/F");
  T1->Branch("phophi",phophi,"phophi[nphotons]/F");
  T1->Branch("phomvaid",phomvaid,"phomvaid[nphotons]/O");
  T1->Branch("phoe1by9",phoe1by9,"phoe1by9[nphotons]/F");
  T1->Branch("phoe9by25",phoe9by25,"phoe9by25[nphotons]/F");
  T1->Branch("photrkiso",photrkiso,"photrkiso[nphotons]/F");
  T1->Branch("phoemiso",phoemiso,"phoemiso[nphotons]/F");
  T1->Branch("phohadiso",phohadiso,"phohadiso[nphotons]/F");
  T1->Branch("phochhadiso",phochhadiso,"phochhadiso[nphotons]/F");
  T1->Branch("phoneuhadiso",phoneuhadiso,"phoneuhadiso[nphotons]/F");
  T1->Branch("phophoiso",phophoiso,"phophoiso[nphotons]/F");
  T1->Branch("phoPUiso",phoPUiso,"phoPUiso[nphotons]/F");
  T1->Branch("phohadbyem",phohadbyem,"phohadbyem[nphotons]/F");
  T1->Branch("phoietaieta",phoietaieta,"phoietaieta[nphotons]/F");
  
  Nevt=0;
  ncnt = 0;
  irunold = -1;
  
}


ExJets::~ExJets()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
ExJets::analyze(const edm::Event& iEvent, const edm::EventSetup& pset) {

  using namespace edm;
  Nevt++;

  irun = iEvent.id().run();
  ilumi = iEvent.luminosityBlock();
  
  ievt = iEvent.id().event();
  
  //if (Nevt%100==1)cout <<"ExJets::analyze "<<Nevt<<" "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  //cout <<"ExJets::analyze "<<Nevt<<" "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  
//  HLTConfigProvider const&  hltConfig_ = hltPrescaleProvider_.hltConfigProvider(); 
  
   wtfact = 1.;

  if(isMC){
  edm::Handle<GenEventInfoProduct>eventinfo ;  
  iEvent.getByToken(tok_wt_,eventinfo) ;
    
    if (eventinfo.isValid()){
      event_weight = eventinfo->weight();
      qscale = eventinfo->qScale();
    }
 
  wtfact *= event_weight;
  }
  
  Handle<VertexCollection> primaryVertices;
  iEvent.getByToken(tok_primaryVertices_, primaryVertices);
  
  edm::Handle<reco::VertexCollection> vertexHandle;
  iEvent.getByToken(tok_primaryVertices_, vertexHandle);
  reco::Vertex vertex = vertexHandle->at(0);
 
  if (primaryVertices.isValid()) {
    int ndofct_org=0;
    int nchict_org=0;
    int nvert_org = 0;
    for (reco::VertexCollection::const_iterator vert=primaryVertices->begin(); vert<primaryVertices->end(); vert++) {
      nvert_org++;
      if (vert->isValid() && !vert->isFake()) {
        if (vert->ndof()>7) {
          ndofct_org++;
          if (vert->normalizedChi2()<5) nchict_org++;
                }
      }
    }
    nprim = min(99,nvert_org) + 100*min(99,ndofct_org) + 10000*min(99,nchict_org);

    nvert = nvert_org;
    nchict = nchict_org;
    ndofct = ndofct_org;
 
  } else { nprim = 0;}

  reco::TrackBase::Point beamPoint(0,0, 0);
  edm::Handle<reco::BeamSpot> beamSpotH;

  iEvent.getByToken(tok_beamspot_, beamSpotH);  //Label("offlineBeamSpot",beamSpotH);
  if (beamSpotH.isValid()){
    beamPoint = beamSpotH->position();
  }

  npu_vert = 0;

  edm::Handle<reco::VertexCompositePtrCandidateCollection> svin;
  iEvent.getByToken(tok_sv,svin);

  if (isMC) {

    edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;
    iEvent.getByToken(pileup_, PupInfo);
    int npu = -1;
    if (PupInfo.isValid()) {
      std::vector<PileupSummaryInfo>::const_iterator PVI;
      for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
                if (PVI->getBunchCrossing()==0) {
                        npu = PVI->getPU_NumInteractions();
                        break;
                }
      }
    }

   npu_vert = npu;

  }//isMC

  Handle<double> Rho_PF;
  
  iEvent.getByToken(tok_Rho_,Rho_PF);
  Rho = *Rho_PF;
  
  const char* variab1;
   
  edm::Handle<edm::TriggerResults> trigRes;
  iEvent.getByToken(triggerBits_, trigRes);
  
  const edm::TriggerNames &names = iEvent.triggerNames(*trigRes);

  edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjects;
  iEvent.getByToken(triggerObjects_, triggerObjects);
  
  edm::Handle<pat::PackedTriggerPrescales> triggerPrescales;
  iEvent.getByToken(triggerPrescales_, triggerPrescales);
   
  // Trigger Selection;
  
  int ihlttrg[nHLTmx+1]= {0};
  int iprescale[nHLTmx]= {0};
  
  for (int jk=0; jk<nHLTmx; jk++) {
    for(unsigned ij = 0; ij<trigRes->size(); ++ij) {
       std::string name = names.triggerName(ij);
 
       variab1 = name.c_str(); 
       
       if (strstr(variab1,hlt_name[jk]) && ((strlen(variab1)-strlen(hlt_name[jk]))<5))
        {
		if ((trigRes->accept(ij))){  
			ihlttrg[jk] = ihlttrg[nHLTmx] = 1;
			iprescale[jk] = hltPrescaleProvider_.prescaleValue(iEvent,pset,name);
			}
        }
     }//ij     
  
    }//jk
 /* 
  if (!isMC) {
    if (ihlttrg[nHLTmx]>0){
			wtfact = 1.0;///compres[ihlt];
		  } else { return ; } 
	}
  */
   trig_value = 1;

  for (int jk=1; jk<(nHLTmx+1); jk++) {
	  if(ihlttrg[nHLTmx-jk]>0) {
		  trig_value+=(1<<jk);
		  }
  }

  for (int ij=0; ij<nHLTmx; ij++) {
    weighttrg[ij] = wtfact;
    if (!isMC) { weighttrg[ij] *=compres[ij];}
	}//ij
  
  vector<triggervar> alltrgobj;
  if (trigRes.isValid()) { 
  
    const char* variab2 ;
 
    alltrgobj.clear(); 
   
   // const edm::TriggerNames &names = iEvent.triggerNames(*trigRes);
    for (pat::TriggerObjectStandAlone obj : *triggerObjects) {
   
      obj.unpackPathNames(names);
      std::vector<std::string> pathNamesAll  = obj.pathNames(false);
      
      for (unsigned ih = 0, n = pathNamesAll.size(); ih < n; ih++) {
		variab2 = pathNamesAll[ih].c_str(); 
		
			for (int jk=0; jk<nHLTmx; jk++) {
				
				if (strstr(variab2,hlt_name[jk]) && (strlen(variab2)-strlen(hlt_name[jk])<5)) {
				
					if(obj.pt()>20 && fabs(obj.eta())<3.0) {
						
						triggervar tmpvec1;
						
						tmpvec1.both = obj.hasPathName( pathNamesAll[ih], true, true );
						tmpvec1.highl  = obj.hasPathName( pathNamesAll[ih], false, true );
						tmpvec1.level1 = obj.hasPathName( pathNamesAll[ih], true, false );
						tmpvec1.trg4v = HepLorentzVector(obj.px(), obj.py(), obj.pz(), obj.energy());
						tmpvec1.prescl = 1;    //triggerPrescales->getPrescaleForIndex(ij);
						tmpvec1.ihlt = jk;
						alltrgobj.push_back(tmpvec1);
						
						}
					}
				}//jk 
		}//ih
	 }
  }
  
  ntrigobjs = alltrgobj.size();
  if(ntrigobjs>njetmx) { ntrigobjs = njetmx; }
  if(alltrgobj.size()>0){
	  for(unsigned int iht=0; iht<(alltrgobj.size()); iht++){
			trigobjpt[iht] = alltrgobj[iht].trg4v.perp();
			trigobjeta[iht] = alltrgobj[iht].trg4v.rapidity();
			trigobjphi[iht] = alltrgobj[iht].trg4v.phi();
			trigobje[iht] = alltrgobj[iht].trg4v.e();
			trigobjHLT[iht] = alltrgobj[iht].highl;
			trigobjL1[iht] = alltrgobj[iht].level1;
			trigobjBoth[iht] = alltrgobj[iht].both;
			trigobjIhlt[iht] = alltrgobj[iht].ihlt;
			if(iht == (njetmx-1)) break;
		}
	  }
  
 
  miset = misphi = -1000 ;
  
  edm::Handle<pat::METCollection> pfmet_ ;
  iEvent.getByToken(tok_mets_,pfmet_) ;
  
  if(pfmet_.isValid()){
	      const pat::MET &met = pfmet_->front();
          miset = met.corPt(); //met.pt();
          misphi = met.corPhi();//met.phi();
          sumEt = met.corSumEt();//sumEt();
          if(isMC){
			  genmiset = met.genMET()->pt();
			  genmisphi = met.genMET()->phi();
			  }
          }

    edm::Handle<edm::View<pat::Jet>> pfjetAK8s;
    edm::Handle<edm::View<pat::Jet>> pfjetAK8s_CHS;
    edm::Handle<edm::View<pat::Jet>> pfjetAK8s_Puppi;
    edm::Handle<reco::GenJetCollection> genjetAK8s;
 
    JetDefinition pfjetAK8Def_CA(cambridge_algorithm,8,E_scheme); // see that jet radius is kept my larger so that all the particles of the initial jet are captured into one final jet
    
    JetDefinition jetDefkT(kt_algorithm,0.4,E_scheme);
    JetDefinition jetDefantikT(antikt_algorithm,0.4,E_scheme);
    JetDefinition jetDefCA(cambridge_algorithm,0.4,E_scheme);
    
    SoftDrop sd(beta,z_cut,0.8);
    SoftDrop sd1(beta1,z_cut,0.8);
    SoftDrop sd2(beta2,z_cut,0.8);
    Filter treeTrimmer(Rsub_trim, SelectorPtFractionMin(fcut_trim));
    Pruner pruner(antikt_algorithm, zcut_prun, Rcut_prun);
    
    edm::Handle<edm::View<pat::Jet>> pfjetAK4s;
    edm::Handle<reco::GenJetCollection> genjetAK4s;
    JetDefinition pfjetAK4Def(antikt_algorithm,0.4,E_scheme);
	
//	edm::Handle<edm::View<pat::PackedGenParticle>> genparticles;
	edm::Handle<std::vector<reco::GenParticle>> genparticles;
	
    npfjetAK8 = 0;
 
	iEvent.getByToken(tok_pfjetAK8s_, pfjetAK8s);	
	if(isMC){
	iEvent.getByToken(tok_genjetAK8s_, genjetAK8s);
	}
	
	 if(pfjetAK8s.isValid()){
	for (unsigned jet = 0; jet< pfjetAK8s->size(); jet++) {
	  
	  HepLorentzVector pfjetAK84v((*pfjetAK8s)[jet].correctedP4("Uncorrected").px(),(*pfjetAK8s)[jet].correctedP4("Uncorrected").py(),(*pfjetAK8s)[jet].correctedP4("Uncorrected").pz(), (*pfjetAK8s)[jet].correctedP4("Uncorrected").energy());
	  HepLorentzVector tmpjetAK84v((*pfjetAK8s)[jet].px(),(*pfjetAK8s)[jet].py(),(*pfjetAK8s)[jet].pz(), (*pfjetAK8s)[jet].energy());
	  
	  double tmprecpt = pfjetAK84v.perp();
	  if(tmprecpt<AK8PtCut) continue;
	  if(tmpjetAK84v.perp()<AK8PtCut) continue;
	  if(abs(pfjetAK84v.rapidity())>maxEta) continue;
 
	  pfjetAK8pt[npfjetAK8] = 	tmprecpt;
	  pfjetAK8y[npfjetAK8] = pfjetAK84v.rapidity();
	  pfjetAK8eta[npfjetAK8] = pfjetAK84v.eta();
	  pfjetAK8phi[npfjetAK8] = pfjetAK84v.phi();
	  pfjetAK8mass[npfjetAK8] = (*pfjetAK8s)[jet].correctedP4("Uncorrected").mass();
							
	  double total_cor =1;
	  
	  jecL1FastAK8->setJetPt(tmprecpt); jecL1FastAK8->setJetA((*pfjetAK8s)[jet].jetArea()); jecL1FastAK8->setRho(*Rho_PF);jecL1FastAK8->setJetEta(pfjetAK84v.eta());
      double corFactorL1Fast = jecL1FastAK8->getCorrection();
      total_cor *= corFactorL1Fast;
      tmprecpt = tmprecpt * corFactorL1Fast;
      pfjetAK8JECL1[npfjetAK8] = corFactorL1Fast;
      
      jecL2RelativeAK8->setJetPt(tmprecpt); jecL2RelativeAK8->setJetEta(pfjetAK84v.eta());
      double corFactorL2Relative = jecL2RelativeAK8->getCorrection();
      total_cor *= corFactorL2Relative ;
      tmprecpt = tmprecpt * corFactorL2Relative;
      pfjetAK8JECL2[npfjetAK8] = corFactorL2Relative;
      
      jecL3AbsoluteAK8->setJetPt(tmprecpt); jecL3AbsoluteAK8->setJetEta(pfjetAK84v.eta());
	  double corFactorL3Absolute = jecL3AbsoluteAK8->getCorrection();
	  total_cor *= corFactorL3Absolute ;
	  tmprecpt = tmprecpt * corFactorL3Absolute;
	  pfjetAK8JECL3[npfjetAK8] = corFactorL3Absolute;
	
	  double corFactorL2L3Residual=1.;
	
	  if(isData){
			jecL2L3ResidualAK8->setJetPt(tmprecpt); jecL2L3ResidualAK8->setJetEta(pfjetAK84v.eta());
			corFactorL2L3Residual = jecL2L3ResidualAK8->getCorrection();
			total_cor*= corFactorL2L3Residual;
  	      	tmprecpt *=corFactorL2L3Residual;
		}
		
	  pfjetAK8JECL2L3[npfjetAK8] = corFactorL2L3Residual;
	  pfjetAK8JEC[npfjetAK8] = total_cor;

      if(isMC){
		  
	  JME::JetResolution resolution_AK8;
	  resolution_AK8 = JME::JetResolution(mPtResoFileAK8.c_str());
	  JME::JetResolutionScaleFactor res_sf_AK8;
	  res_sf_AK8 = JME::JetResolutionScaleFactor(mPtSFFileAK8.c_str());
	 
	  JME::JetParameters parameters_5 = {{JME::Binning::JetPt, tmprecpt}, {JME::Binning::JetEta, pfjetAK84v.eta()}, {JME::Binning::Rho, *Rho_PF}};
	  double rp_AK8 = resolution_AK8.getResolution(parameters_5);
	  double gaus_rp_AK8 = gRandom->Gaus(0.,rp_AK8);
	  double sf_AK8 = res_sf_AK8.getScaleFactor(parameters_5, Variation::NOMINAL);
	 
	  bool match_AK8 = false;
      int match_gen_AK8 = -1;
		        
      for (unsigned get = 0; get<(genjetAK8s->size()); get++) {
		HepLorentzVector genjet8v((*genjetAK8s)[get].px(),(*genjetAK8s)[get].py(),(*genjetAK8s)[get].pz(), (*genjetAK8s)[get].energy());
		 if((delta2R(pfjetAK84v.rapidity(),pfjetAK84v.phi(),genjet8v.rapidity(),genjet8v.phi()) < (0.5*0.8)) &&(fabs(tmprecpt-genjet8v.perp())<(3*fabs(rp_AK8)*tmprecpt))){
			match_AK8 = true;
			match_gen_AK8 = get;
			break;
			}
		}

		if(match_AK8&&(match_gen_AK8>=0)){
			pfjetAK8reso[npfjetAK8] = (sf_AK8-1.)*(tmprecpt-(*genjetAK8s)[match_gen_AK8].pt())*1./tmprecpt;
				}else{
	  
			pfjetAK8reso[npfjetAK8] = sqrt(max(0.,(sf_AK8*sf_AK8-1))) * gaus_rp_AK8;
		}
	   
	   }//isMC
	  
	  pfjetAK8tightID[npfjetAK8] = false;

	  int NumConst = (*pfjetAK8s)[jet].chargedMultiplicity() + (*pfjetAK8s)[jet].neutralMultiplicity();
	  double eta = pfjetAK84v.eta();
	  float CEMF = (*pfjetAK8s)[jet].chargedEmEnergyFraction();
	  float CHF =  (*pfjetAK8s)[jet].chargedHadronEnergyFraction();
	  float NEMF = (*pfjetAK8s)[jet].neutralEmEnergyFraction();
	  float NHF = (*pfjetAK8s)[jet].neutralHadronEnergyFraction();
	  int CHM = (*pfjetAK8s)[jet].chargedHadronMultiplicity();
	  int NumNeutralParticle = (*pfjetAK8s)[jet].neutralMultiplicity();
	  pfjetAK8tightID[npfjetAK8] = ((abs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 )) || ((abs(eta)>2.6 && abs(eta)<=2.7 && CEMF<0.8 && CHM>0 && NEMF<0.99 && NHF < 0.9 )) || (NEMF>0.02 && NEMF<0.99 && NumNeutralParticle>2 && abs(eta)>2.7 && abs(eta)<=3.0 ) || (NEMF<0.90 && NHF>0.2 && NumNeutralParticle>10 && abs(eta)>3.0 );
	  
	  std::vector<reco::CandidatePtr> daught((*pfjetAK8s)[jet].daughterPtrVector());
      std::sort(daught.begin(), daught.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2)
      { return p1->pt() > p2->pt(); });

	  vector <fastjet::PseudoJet> fjInputs;
	  fjInputs.resize(0);

	  for (unsigned int i2 = 0; i2< daught.size(); ++i2) {
		  
		  PseudoJet psjet ;
		  psjet = PseudoJet( (*daught[i2]).px(),(*daught[i2]).py(),(*daught[i2]).pz(),(*daught[i2]).energy() );
		  psjet.set_user_index(i2);
          fjInputs.push_back(psjet);
	
		  }
	  
	  daught.clear();
	  
	  vector <fastjet::PseudoJet> sortedJets;
	  fastjet::ClusterSequence clustSeq(fjInputs, pfjetAK8Def_CA);
	  fjInputs.clear();
	  sortedJets    = sorted_by_pt(clustSeq.inclusive_jets());
	  
	  if(sortedJets.size()>0){
			pfjetAK8sdmass[npfjetAK8] = sd(sortedJets[0]).m();		
			pfjetAK8sdmass1[npfjetAK8] = sd1(sortedJets[0]).m();		
			pfjetAK8sdmass2[npfjetAK8] = sd2(sortedJets[0]).m();		
			pfjetAK8trimmass[npfjetAK8] = treeTrimmer(sortedJets[0]).m();
			pfjetAK8prunmass[npfjetAK8] =	pruner(sortedJets[0]).m();
	  }
	  sortedJets.clear();
	
	  npfjetAK8++;	
	  if(npfjetAK8 >= njetmxAK8) { break;}
	  
	}
  }
  
    npfjetAK8_CHS = 0;
 
	iEvent.getByToken(tok_pfjetAK8s_CHS_, pfjetAK8s_CHS);	
	
	if(pfjetAK8s_CHS.isValid()){
	for (unsigned jet = 0; jet< pfjetAK8s_CHS->size(); jet++) {
	  
	  HepLorentzVector pfjetAK84v_CHS((*pfjetAK8s_CHS)[jet].px(),(*pfjetAK8s_CHS)[jet].py(),(*pfjetAK8s_CHS)[jet].pz(), (*pfjetAK8s_CHS)[jet].energy());
	  
	  double tmprecpt = pfjetAK84v_CHS.perp();
	  if(tmprecpt<AK8PtCut) continue;
	  if(pfjetAK84v_CHS.perp()<AK8PtCut) continue;
	  if(abs(pfjetAK84v_CHS.rapidity())>maxEta) continue;
 
	  pfjetAK8pt_CHS[npfjetAK8_CHS] = 	tmprecpt;
	  pfjetAK8y_CHS[npfjetAK8_CHS] = pfjetAK84v_CHS.rapidity();
	  pfjetAK8eta_CHS[npfjetAK8_CHS] = pfjetAK84v_CHS.eta();
	  pfjetAK8phi_CHS[npfjetAK8_CHS] = pfjetAK84v_CHS.phi();
	  pfjetAK8mass_CHS[npfjetAK8_CHS] = (*pfjetAK8s_CHS)[jet].mass();
							
      if(isMC){
		  
	  JME::JetResolution resolution_AK8;
	  resolution_AK8 = JME::JetResolution(mPtResoFileAK8CHS.c_str());
	  JME::JetResolutionScaleFactor res_sf_AK8;
	  res_sf_AK8 = JME::JetResolutionScaleFactor(mPtSFFileAK8CHS.c_str());
	 
	  JME::JetParameters parameters_5 = {{JME::Binning::JetPt, tmprecpt}, {JME::Binning::JetEta, pfjetAK84v_CHS.eta()}, {JME::Binning::Rho, *Rho_PF}};
	  double rp_AK8 = resolution_AK8.getResolution(parameters_5);
	  double gaus_rp_AK8 = gRandom->Gaus(0.,rp_AK8);
	  double sf_AK8 = res_sf_AK8.getScaleFactor(parameters_5, Variation::NOMINAL);
	 
	  bool match_AK8 = false;
      int match_gen_AK8 = -1;
		        
      for (unsigned get = 0; get<(genjetAK8s->size()); get++) {
		HepLorentzVector genjet8v((*genjetAK8s)[get].px(),(*genjetAK8s)[get].py(),(*genjetAK8s)[get].pz(), (*genjetAK8s)[get].energy());
		 if((delta2R(pfjetAK84v_CHS.rapidity(),pfjetAK84v_CHS.phi(),genjet8v.rapidity(),genjet8v.phi()) < (0.5*0.8)) &&(fabs(tmprecpt-genjet8v.perp())<(3*fabs(rp_AK8)*tmprecpt))){
			match_AK8 = true;
			match_gen_AK8 = get;
			break;
			}
		}

		if(match_AK8&&(match_gen_AK8>=0)){
			pfjetAK8reso_CHS[npfjetAK8_CHS] = (sf_AK8-1.)*(tmprecpt-(*genjetAK8s)[match_gen_AK8].pt())*1./tmprecpt;
				}else{
	  
			pfjetAK8reso_CHS[npfjetAK8_CHS] = sqrt(max(0.,(sf_AK8*sf_AK8-1))) * gaus_rp_AK8;
		}
	   
	   }//isMC
	  
	  pfjetAK8tightID_CHS[npfjetAK8_CHS] = false;
	
	  int NumConst = (*pfjetAK8s_CHS)[jet].chargedMultiplicity() + (*pfjetAK8s_CHS)[jet].neutralMultiplicity();
	  double eta = pfjetAK84v_CHS.eta();
	  float CEMF = (*pfjetAK8s_CHS)[jet].chargedEmEnergyFraction();
	  float CHF =  (*pfjetAK8s_CHS)[jet].chargedHadronEnergyFraction();
	  float NEMF = (*pfjetAK8s_CHS)[jet].neutralEmEnergyFraction();
	  float NHF = (*pfjetAK8s_CHS)[jet].neutralHadronEnergyFraction();
	  int CHM = (*pfjetAK8s_CHS)[jet].chargedHadronMultiplicity();
	  int NumNeutralParticle = (*pfjetAK8s_CHS)[jet].neutralMultiplicity();
	  pfjetAK8tightID_CHS[npfjetAK8_CHS] = ((abs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 )) || ((abs(eta)>2.6 && abs(eta)<=2.7 && CEMF<0.8 && CHM>0 && NEMF<0.99 && NHF < 0.9 )) || (NEMF>0.02 && NEMF<0.99 && NumNeutralParticle>2 && abs(eta)>2.7 && abs(eta)<=3.0 ) || (NEMF<0.90 && NHF>0.2 && NumNeutralParticle>10 && abs(eta)>3.0 );
	 
	  std::vector<reco::CandidatePtr> daught((*pfjetAK8s_CHS)[jet].daughterPtrVector());
      std::sort(daught.begin(), daught.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2)
      { return p1->pt() > p2->pt(); });

	  vector <fastjet::PseudoJet> fjInputs;
	  fjInputs.resize(0);

	  for (unsigned int i2 = 0; i2< daught.size(); ++i2) {
		  
		  PseudoJet psjet ;
		  psjet = PseudoJet( (*daught[i2]).px(),(*daught[i2]).py(),(*daught[i2]).pz(),(*daught[i2]).energy() );
		  psjet.set_user_index(i2);
          fjInputs.push_back(psjet);
	
		  }
	  
	  daught.clear();
	  
	  vector <fastjet::PseudoJet> sortedJets;
	  fastjet::ClusterSequence clustSeq(fjInputs, pfjetAK8Def_CA);
	  fjInputs.clear();
	  sortedJets    = sorted_by_pt(clustSeq.inclusive_jets());
	  
	  if(sortedJets.size()>0){
			pfjetAK8sdmass_CHS[npfjetAK8_CHS] = sd(sortedJets[0]).m();	
			pfjetAK8sdmass1_CHS[npfjetAK8_CHS] = sd1(sortedJets[0]).m();		
			pfjetAK8sdmass2_CHS[npfjetAK8_CHS] = sd2(sortedJets[0]).m();			
			pfjetAK8trimmass_CHS[npfjetAK8_CHS] = treeTrimmer(sortedJets[0]).m();
			pfjetAK8prunmass_CHS[npfjetAK8_CHS] =	pruner(sortedJets[0]).m();
	  }
	  sortedJets.clear();
	
	  npfjetAK8_CHS++;	
	  if(npfjetAK8_CHS >= njetmxAK8) { break;}
	  
	}
  }

	
    npfjetAK8_Puppi = 0;
 
	iEvent.getByToken(tok_pfjetAK8s_Puppi_, pfjetAK8s_Puppi);	
	
	if(pfjetAK8s_Puppi.isValid()){
	for (unsigned jet = 0; jet< pfjetAK8s_Puppi->size(); jet++) {
	  
	  HepLorentzVector pfjetAK84v_Puppi((*pfjetAK8s_Puppi)[jet].px(),(*pfjetAK8s_Puppi)[jet].py(),(*pfjetAK8s_Puppi)[jet].pz(), (*pfjetAK8s_Puppi)[jet].energy());
	  
	  double tmprecpt = pfjetAK84v_Puppi.perp();
	  if(tmprecpt<AK8PtCut) continue;
	  if(pfjetAK84v_Puppi.perp()<AK8PtCut) continue;
	  if(abs(pfjetAK84v_Puppi.rapidity())>maxEta) continue;
 
	  pfjetAK8pt_Puppi[npfjetAK8_Puppi] = 	tmprecpt;
	  pfjetAK8y_Puppi[npfjetAK8_Puppi] = pfjetAK84v_Puppi.rapidity();
	  pfjetAK8eta_Puppi[npfjetAK8_Puppi] = pfjetAK84v_Puppi.eta();
	  pfjetAK8phi_Puppi[npfjetAK8_Puppi] = pfjetAK84v_Puppi.phi();
	  pfjetAK8mass_Puppi[npfjetAK8_Puppi] = (*pfjetAK8s_Puppi)[jet].mass();
							
      if(isMC){
		  
	  JME::JetResolution resolution_AK8;
	  resolution_AK8 = JME::JetResolution(mPtResoFileAK8Puppi.c_str());
	  JME::JetResolutionScaleFactor res_sf_AK8;
	  res_sf_AK8 = JME::JetResolutionScaleFactor(mPtSFFileAK8Puppi.c_str());
	 
	  JME::JetParameters parameters_5 = {{JME::Binning::JetPt, tmprecpt}, {JME::Binning::JetEta, pfjetAK84v_Puppi.eta()}, {JME::Binning::Rho, *Rho_PF}};
	  double rp_AK8 = resolution_AK8.getResolution(parameters_5);
	  double gaus_rp_AK8 = gRandom->Gaus(0.,rp_AK8);
	  double sf_AK8 = res_sf_AK8.getScaleFactor(parameters_5, Variation::NOMINAL);
	 
	  bool match_AK8 = false;
      int match_gen_AK8 = -1;
		        
      for (unsigned get = 0; get<(genjetAK8s->size()); get++) {
		HepLorentzVector genjet8v((*genjetAK8s)[get].px(),(*genjetAK8s)[get].py(),(*genjetAK8s)[get].pz(), (*genjetAK8s)[get].energy());
		 if((delta2R(pfjetAK84v_Puppi.rapidity(),pfjetAK84v_Puppi.phi(),genjet8v.rapidity(),genjet8v.phi()) < (0.5*0.8)) &&(fabs(tmprecpt-genjet8v.perp())<(3*fabs(rp_AK8)*tmprecpt))){
			match_AK8 = true;
			match_gen_AK8 = get;
			break;
			}
		}

		if(match_AK8&&(match_gen_AK8>=0)){
			pfjetAK8reso_Puppi[npfjetAK8_Puppi] = (sf_AK8-1.)*(tmprecpt-(*genjetAK8s)[match_gen_AK8].pt())*1./tmprecpt;
				}else{
	  
			pfjetAK8reso_Puppi[npfjetAK8_Puppi] = sqrt(max(0.,(sf_AK8*sf_AK8-1))) * gaus_rp_AK8;
		}
	   
	   }//isMC
	  
	  pfjetAK8tightID_Puppi[npfjetAK8_Puppi] = false;
	
	  int NumConst = (*pfjetAK8s_Puppi)[jet].chargedMultiplicity() + (*pfjetAK8s_Puppi)[jet].neutralMultiplicity();
	  double eta = pfjetAK84v_Puppi.eta();
	  float CEMF = (*pfjetAK8s_Puppi)[jet].chargedEmEnergyFraction();
	  float CHF =  (*pfjetAK8s_Puppi)[jet].chargedHadronEnergyFraction();
	  float NEMF = (*pfjetAK8s_Puppi)[jet].neutralEmEnergyFraction();
	  float NHF = (*pfjetAK8s_Puppi)[jet].neutralHadronEnergyFraction();
	  int CHM = (*pfjetAK8s_Puppi)[jet].chargedHadronMultiplicity();
	  int NumNeutralParticle = (*pfjetAK8s_Puppi)[jet].neutralMultiplicity();
	  pfjetAK8tightID_Puppi[npfjetAK8_Puppi] = ((abs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 )) || ((abs(eta)>2.6 && abs(eta)<=2.7 && CEMF<0.8 && NEMF<0.99 && NHF < 0.9 )) || (NEMF<0.99 && abs(eta)>2.7 && abs(eta)<=3.0 ) || (NEMF<0.90 && NHF>0.02 && NumNeutralParticle>2 && NumNeutralParticle<15 && abs(eta)>3.0 );
	 
	  
	  std::vector<reco::CandidatePtr> daught((*pfjetAK8s_Puppi)[jet].daughterPtrVector());
      std::sort(daught.begin(), daught.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2)
      { return p1->pt() > p2->pt(); });

	  vector <fastjet::PseudoJet> fjInputs;
	  fjInputs.resize(0);

	  for (unsigned int i2 = 0; i2< daught.size(); ++i2) {
		  
		  PseudoJet psjet ;
		  psjet = PseudoJet( (*daught[i2]).px(),(*daught[i2]).py(),(*daught[i2]).pz(),(*daught[i2]).energy() );
		  psjet.set_user_index(i2);
          fjInputs.push_back(psjet);
	
		  }
	  
	  daught.clear();
	  
	  vector <fastjet::PseudoJet> sortedJets;
	  fastjet::ClusterSequence clustSeq(fjInputs, pfjetAK8Def_CA);
	  fjInputs.clear();
	  sortedJets    = sorted_by_pt(clustSeq.inclusive_jets());
	  
	  if(sortedJets.size()>0){
			pfjetAK8sdmass_Puppi[npfjetAK8_Puppi] = sd(sortedJets[0]).m();	
			pfjetAK8sdmass1_Puppi[npfjetAK8_Puppi] = sd1(sortedJets[0]).m();	
			pfjetAK8sdmass2_Puppi[npfjetAK8_Puppi] = sd2(sortedJets[0]).m();		
			pfjetAK8trimmass_Puppi[npfjetAK8_Puppi] = treeTrimmer(sortedJets[0]).m();
			pfjetAK8prunmass_Puppi[npfjetAK8_Puppi] =	pruner(sortedJets[0]).m();
	  }
	  sortedJets.clear();
	
	  npfjetAK8_Puppi++;	
	  if(npfjetAK8_Puppi >= njetmxAK8) { break;}
	  
	}
  }
	
	npfjetAK4 = 0;
	iEvent.getByToken(tok_pfjetAK4s_, pfjetAK4s);
	if(isMC){
	iEvent.getByToken(tok_genjetAK4s_, genjetAK4s);
	}
	
	for (unsigned jet = 0; jet< pfjetAK4s->size(); jet++) {
	  
	  HepLorentzVector pfjetAK44v((*pfjetAK4s)[jet].correctedP4("Uncorrected").px(),(*pfjetAK4s)[jet].correctedP4("Uncorrected").py(),(*pfjetAK4s)[jet].correctedP4("Uncorrected").pz(), (*pfjetAK4s)[jet].correctedP4("Uncorrected").energy());
  
	  double tmprecpt = pfjetAK44v.perp();
 
	  if(tmprecpt<minPt) continue;
	  if(abs(pfjetAK44v.eta())>maxEta) continue;
 
	  pfjetAK4pt[npfjetAK4] = 	tmprecpt;
	  pfjetAK4eta[npfjetAK4] = 	pfjetAK44v.eta();
	  pfjetAK4y[npfjetAK4] = pfjetAK44v.rapidity();
	  pfjetAK4phi[npfjetAK4] = pfjetAK44v.phi();
	  pfjetAK4mass[npfjetAK4] = (*pfjetAK4s)[jet].correctedP4("Uncorrected").mass();
	  
	  double total_cor =1;
	  
	  jecL1FastAK4->setJetPt(tmprecpt); jecL1FastAK4->setJetA((*pfjetAK4s)[jet].jetArea()); jecL1FastAK4->setRho(*Rho_PF);jecL1FastAK4->setJetEta(pfjetAK44v.eta());
      double corFactorL1Fast = jecL1FastAK4->getCorrection();
      total_cor*= corFactorL1Fast;
      tmprecpt = tmprecpt * corFactorL1Fast;
      
      jecL2RelativeAK4->setJetPt(tmprecpt); jecL2RelativeAK4->setJetEta(pfjetAK44v.eta());
      double corFactorL2Relative = jecL2RelativeAK4->getCorrection();
      total_cor*= corFactorL2Relative ;
      tmprecpt = tmprecpt * corFactorL2Relative;
      
      jecL3AbsoluteAK4->setJetPt(tmprecpt); jecL3AbsoluteAK4->setJetEta(pfjetAK44v.eta());
      double corFactorL3Absolute = jecL3AbsoluteAK4->getCorrection();
      total_cor*= corFactorL3Absolute ;
      tmprecpt = tmprecpt * corFactorL3Absolute;

      double corFactorL2L3Residual=1.;
	
	  if(!isMC){
		  
		  jecL2L3ResidualAK4->setJetPt(tmprecpt); jecL2L3ResidualAK4->setJetEta(pfjetAK44v.eta());
    
		  corFactorL2L3Residual = jecL2L3ResidualAK4->getCorrection();
		  total_cor*= corFactorL2L3Residual;
  	      tmprecpt *=corFactorL2L3Residual;
		}
		
	  pfjetAK4JEC[npfjetAK4] = total_cor;
	  
	  pfjetAK4JECL1[npfjetAK4] = corFactorL1Fast;
	  pfjetAK4JECL2[npfjetAK4] = corFactorL2Relative;
	  pfjetAK4JECL3[npfjetAK4] = corFactorL3Absolute;
	  pfjetAK4JECL2L3[npfjetAK4] = corFactorL2L3Residual;
	  
	  if(isMC){

	  JME::JetResolution resolution_AK4;
	  resolution_AK4 = JME::JetResolution(mPtResoFileAK4.c_str());
	  JME::JetResolutionScaleFactor res_sf_AK4;
	  res_sf_AK4 = JME::JetResolutionScaleFactor(mPtSFFileAK4.c_str());
	 
	  JME::JetParameters parameters_5 = {{JME::Binning::JetPt, tmprecpt}, {JME::Binning::JetEta, pfjetAK44v.eta()}, {JME::Binning::Rho, *Rho_PF}};
	  double rp_AK4 = resolution_AK4.getResolution(parameters_5);
	  double gaus_rp_AK4 = gRandom->Gaus(0.,rp_AK4);
	  double sf_AK4 = res_sf_AK4.getScaleFactor(parameters_5, Variation::NOMINAL);
	  
	  bool match_AK4 = false;
      int match_gen_AK4 = -1;
           
      for (unsigned get = 0; get<(genjetAK4s->size()); get++) {
		HepLorentzVector genjet4v((*genjetAK4s)[get].px(),(*genjetAK4s)[get].py(),(*genjetAK4s)[get].pz(), (*genjetAK4s)[get].energy());
		 if((delta2R(pfjetAK44v.rapidity(),pfjetAK44v.phi(),genjet4v.rapidity(),genjet4v.phi()) < (0.5*0.4)) &&(fabs(tmprecpt-genjet4v.perp())<(3*fabs(rp_AK4)*tmprecpt))){
			match_AK4 = true;
			match_gen_AK4 = get;
			break;
			}
		}

		if(match_AK4&&(match_gen_AK4>=0)){
			pfjetAK4reso[npfjetAK4] = (sf_AK4-1.)*(tmprecpt-(*genjetAK4s)[match_gen_AK4].pt())*1./tmprecpt;
			}else{
			pfjetAK4reso[npfjetAK4] = sqrt(max(0.,(sf_AK4*sf_AK4-1))) * gaus_rp_AK4;
		}
		
	  }//isMC
	
	  pfjetAK4tightID[npfjetAK4] = false;
	  
	  int NumConst = (*pfjetAK4s)[jet].chargedMultiplicity() + (*pfjetAK4s)[jet].neutralMultiplicity();
	  double eta = pfjetAK44v.eta();
	  float CEMF = (*pfjetAK4s)[jet].chargedEmEnergyFraction();
	  float CHF =  (*pfjetAK4s)[jet].chargedHadronEnergyFraction();
	  float NEMF = (*pfjetAK4s)[jet].neutralEmEnergyFraction();
	  float NHF = (*pfjetAK4s)[jet].neutralHadronEnergyFraction();
	  int CHM = (*pfjetAK4s)[jet].chargedHadronMultiplicity();
	  int NumNeutralParticle = (*pfjetAK4s)[jet].neutralMultiplicity();
	  pfjetAK4tightID[npfjetAK4] = ((abs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 )) || ((abs(eta)>2.6 && abs(eta)<=2.7 && CEMF<0.8 && CHM>0 && NEMF<0.99 && NHF < 0.9 )) || (NEMF>0.02 && NEMF<0.99 && NumNeutralParticle>2 && abs(eta)>2.7 && abs(eta)<=3.0 ) || (NEMF<0.90 && NHF>0.2 && NumNeutralParticle>10 && abs(eta)>3.0 );
	  
	 
      vector <fastjet::PseudoJet> fjInputs;
	  fjInputs.resize(0);

	  std::vector<reco::CandidatePtr> daught((*pfjetAK4s)[jet].daughterPtrVector());
      std::sort(daught.begin(), daught.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2)
      { return p1->pt() > p2->pt(); });

	  for (unsigned int i2 = 0; i2< daught.size(); ++i2) {
		  PseudoJet psjet ;
		  psjet = PseudoJet( (*daught[i2]).px(),(*daught[i2]).py(),(*daught[i2]).pz(),(*daught[i2]).energy() );
		  fjInputs.push_back(psjet);
	  }
	  
	  vector <fastjet::PseudoJet> sortedJets;
	  fastjet::ClusterSequence clustSeq(fjInputs, pfjetAK4Def);
	  fjInputs.clear();
	  sortedJets    = sorted_by_pt(clustSeq.inclusive_jets());
	  
	  if(sortedJets.size()>0){
		PseudoJet sd_jet = sd(sortedJets[0]);
		pfjetAK4sdmass[npfjetAK4] = sd_jet.m();
	  }
	  sortedJets.clear();
	  
   	  npfjetAK4++;	
	  if(npfjetAK4 >= njetmx) { break;}
	}


    if(isMC){


	edm::Handle<reco::GenMETCollection> genmet_ ;
    iEvent.getByToken(tok_genmets_,genmet_) ;
  
     if(genmet_.isValid()){
		 genmiset = genmet_->begin()->et();
		 genmisphi = genmet_->begin()->phi();
	 }

    ngenjetAK8 = 0;
//	iEvent.getByToken(tok_genjetAK8s_, genjetAK8s);

	for(unsigned gjet = 0; gjet<genjetAK8s->size(); gjet++)	{

        HepLorentzVector genjetAK84v((*genjetAK8s)[gjet].px(),(*genjetAK8s)[gjet].py(),(*genjetAK8s)[gjet].pz(), (*genjetAK8s)[gjet].energy());
		if(genjetAK84v.perp()<minPt) continue;
		if(abs(genjetAK84v.rapidity())>maxgenEta) continue;

	    genjetAK8pt[ngenjetAK8] = genjetAK84v.perp();
	    genjetAK8y[ngenjetAK8] = genjetAK84v.rapidity();
	    genjetAK8phi[ngenjetAK8] = genjetAK84v.phi();
		genjetAK8mass[ngenjetAK8] = (*genjetAK8s)[gjet].mass();
		
		vector <fastjet::PseudoJet> fjInputs;
	    fjInputs.resize(0);

		std::vector<reco::CandidatePtr> daught((*genjetAK8s)[gjet].daughterPtrVector());
	    
	    std::sort(daught.begin(), daught.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2)
            { return p1->pt() > p2->pt(); }); 
     
	
        if(daught.size()>0){    
		for (unsigned int i2 = 0; i2< daught.size(); ++i2) {
   
			PseudoJet psjet ;
		    psjet = PseudoJet( (*daught[i2]).px(),(*daught[i2]).py(),(*daught[i2]).pz(),(*daught[i2]).energy() );
		    psjet.set_user_index(i2);
         	fjInputs.push_back(psjet);
             
        } //i2

	
	    vector <fastjet::PseudoJet> sortedJets;
	    fastjet::ClusterSequence clustSeq(fjInputs, pfjetAK8Def_CA);
	    fjInputs.clear();
	    sortedJets    = sorted_by_pt(clustSeq.inclusive_jets());
	  
	    if(sortedJets.size()>0){
//			PseudoJet sd_jet = sd(sortedJets[0]);
			genjetAK8sdmass[ngenjetAK8] = sd(sortedJets[0]).m();
			genjetAK8sdmass1[ngenjetAK8] = sd1(sortedJets[0]).m();
			genjetAK8sdmass2[ngenjetAK8] = sd2(sortedJets[0]).m();		
			genjetAK8trimmass[ngenjetAK8] = treeTrimmer(sortedJets[0]).m();
			genjetAK8prunmass[ngenjetAK8] =	pruner(sortedJets[0]).m();
		}
		sortedJets.clear();
		
		} 
		
	    if (++ngenjetAK8>=njetmx) break;
	    
		}

	ngenjetAK4 = 0;
//	iEvent.getByToken(tok_genjetAK4s_, genjetAK4s);
//	cout<<"genjetAK4s->size() "<<genjetAK4s->size()<<endl;

	for(unsigned gjet = 0; gjet<genjetAK4s->size(); gjet++)	{

        HepLorentzVector genjetAK44v((*genjetAK4s)[gjet].px(),(*genjetAK4s)[gjet].py(),(*genjetAK4s)[gjet].pz(), (*genjetAK4s)[gjet].energy());
		if(genjetAK44v.perp()<minPt) continue;
		if(abs(genjetAK44v.rapidity())>maxgenEta) continue;

	    genjetAK4pt[ngenjetAK4] = genjetAK44v.perp();
	    genjetAK4y[ngenjetAK4] = genjetAK44v.rapidity();
	    genjetAK4phi[ngenjetAK4] = genjetAK44v.phi();
	    genjetAK4mass[ngenjetAK4] = (*genjetAK4s)[gjet].mass();
	
		if (++ngenjetAK4>=njetmx) break;
		}
	
	 ngenparticles = 0;
	
	iEvent.getByToken(tok_genparticles_,genparticles);
	if(genparticles.isValid()){
		
		for(unsigned ig=0; ig<(genparticles->size()); ig++){

		    if(!(((*genparticles)[ig].status()==1)||((*genparticles)[ig].status()==22)||((*genparticles)[ig].status()==23))) continue;
//		    if(!((*genparticles)[ig].isHardProcess())) continue;
		    
		    const Candidate * mom = (*genparticles)[ig].mother();

		    genpartstatus[ngenparticles] = (*genparticles)[ig].status();
		    genpartpdg[ngenparticles] = (*genparticles)[ig].pdgId();
		    genpartmompdg[ngenparticles] = mom->pdgId();//(*genparticles)[ig].
//		    genpartmomid[ngenparticles] = *mom;//->Id();  "?numberOfMothers>0?motherRef(0).key():-1
		    genpartdaugno[ngenparticles] = (*genparticles)[ig].numberOfDaughters();
		    genpartfromhard[ngenparticles] = (*genparticles)[ig].isHardProcess();
		    genpartfromhardbFSR[ngenparticles] = (*genparticles)[ig].fromHardProcessBeforeFSR();
		    genpartisLastCopyBeforeFSR[ngenparticles] = (*genparticles)[ig].isLastCopyBeforeFSR();
		    genpartisPromptFinalState[ngenparticles] = (*genparticles)[ig].isPromptFinalState();
		    genpartpt[ngenparticles] = (*genparticles)[ig].pt();
		    genparteta[ngenparticles] = (*genparticles)[ig].eta();
		    genpartphi[ngenparticles] = (*genparticles)[ig].phi();
		    genpartm[ngenparticles] = (*genparticles)[ig].mass();
		    genpartq[ngenparticles] = (*genparticles)[ig].charge();
		    
			ngenparticles++;
			if(ngenparticles>=npartmx) break;
			}
		
		vector <fastjet::PseudoJet> fjInputs;
	    fjInputs.resize(0);

		for(unsigned ig=0; ig<(genparticles->size()); ig++){
			
			if(!((*genparticles)[ig].status()==1)) continue;
			if(abs((*genparticles)[ig].pdgId())==12 || abs((*genparticles)[ig].pdgId())==14 || abs((*genparticles)[ig].pdgId())==16) continue;
			if((*genparticles)[ig].pt()<1 || fabs((*genparticles)[ig].eta())>5) continue;
			
			PseudoJet psjet ;
		    psjet = PseudoJet( (*genparticles)[ig].px(),(*genparticles)[ig].py(),(*genparticles)[ig].pz(),(*genparticles)[ig].energy() );
         	fjInputs.push_back(psjet);
			
			}

		vector <fastjet::PseudoJet> sortedJets_antikt;
		vector <fastjet::PseudoJet> sortedJets_kt;
		vector <fastjet::PseudoJet> sortedJets_ca;
		
	    fastjet::ClusterSequence clustSeq_antikt(fjInputs, jetDefantikT);
	    sortedJets_antikt = sorted_by_pt(clustSeq_antikt.inclusive_jets());
	    
	    fastjet::ClusterSequence clustSeq_kt(fjInputs, jetDefkT);
	    sortedJets_kt = sorted_by_pt(clustSeq_kt.inclusive_jets());
	    
	    fastjet::ClusterSequence clustSeq_ca(fjInputs, jetDefCA);
	    sortedJets_ca = sorted_by_pt(clustSeq_ca.inclusive_jets());

		fjInputs.clear();

		ngenjetantikt = 0;
		
		for(unsigned ijet=0; ijet<sortedJets_antikt.size(); ijet++){
			
			if(sortedJets_antikt[ijet].perp()<15 || fabs(sortedJets_antikt[ijet].rapidity())>5) continue;
			
			genjetantiktpt[ngenjetantikt] = sortedJets_antikt[ijet].perp();
			genjetantikty[ngenjetantikt] = sortedJets_antikt[ijet].rapidity();
			genjetantiktphi[ngenjetantikt] = sortedJets_antikt[ijet].phi();
			genjetantiktmass[ngenjetantikt] = sortedJets_antikt[ijet].m();
		
			if(++ngenjetantikt>=njetmx) break;
		}
		
		ngenjetkt = 0;
		
		for(unsigned ijet=0; ijet<sortedJets_kt.size(); ijet++){
			
			if(sortedJets_kt[ijet].perp()<15 || fabs(sortedJets_kt[ijet].rapidity())>5) continue;
			
			genjetktpt[ngenjetkt] = sortedJets_kt[ijet].perp();
			genjetkty[ngenjetkt] = sortedJets_kt[ijet].rapidity();
			genjetktphi[ngenjetkt] = sortedJets_kt[ijet].phi();
			genjetktmass[ngenjetkt] = sortedJets_kt[ijet].m();
			
			if(++ngenjetkt>=njetmx) break;
		}
		
		ngenjetca = 0;
		
		for(unsigned ijet=0; ijet<sortedJets_ca.size(); ijet++){
			
			if(sortedJets_ca[ijet].perp()<15 || fabs(sortedJets_ca[ijet].rapidity())>5) continue;
			
			genjetcapt[ngenjetca] = sortedJets_ca[ijet].perp();
			genjetcay[ngenjetca] = sortedJets_ca[ijet].rapidity();
			genjetcaphi[ngenjetca] = sortedJets_ca[ijet].phi();
			genjetcamass[ngenjetca] = sortedJets_ca[ijet].m();
			
			if(++ngenjetca>=njetmx) break;
			
		}
		
		}
	
				
  }//isMC
   
  nmuons = 0; 
  edm::Handle<edm::View<pat::Muon>> muons;
  iEvent.getByToken(tok_muons_, muons);
  if (muons.isValid() && muons->size()>0) {
    edm::View<pat::Muon>::const_iterator muon1;
    for( muon1 = muons->begin(); muon1 < muons->end(); muon1++ ) {
      if ((!muon1->isTrackerMuon()) && (!muon1->isGlobalMuon()) && (!muon1->isStandAloneMuon())) continue;
      
      TrackRef trkglb =muon1->globalTrack();

      if ((!muon1->isGlobalMuon())) {
        if (muon1->isTrackerMuon()) {
          trkglb =muon1->innerTrack();
        } else {
          trkglb =muon1->outerTrack();
        }
      }
      
      TrackRef trktrk =muon1->innerTrack();

      if ((!muon1->isTrackerMuon())) {
        if (muon1->isGlobalMuon()) {
          trktrk =muon1->globalTrack();
        } else {
          trktrk =muon1->outerTrack();
        }
      }

      if (trktrk->pt()<3.0) continue;
      
      muonisPF[nmuons] = muon1->isPFMuon();
      muonisGL[nmuons] = muon1->isGlobalMuon();
      muonisTRK[nmuons] = muon1->isTrackerMuon();
      
      muonecal[nmuons] = (muon1->calEnergy()).em;
      muonhcal[nmuons] = (muon1->calEnergy()).had;
      muonemiso[nmuons] = (muon1->isolationR03()).emEt;
      muonhadiso[nmuons] = (muon1->isolationR03()).hadEt;
      muontkpt03[nmuons] = (muon1->isolationR03()).sumPt;
      muontkpt05[nmuons] = (muon1->isolationR05()).sumPt;
      muonposmatch[nmuons] = (muon1->combinedQuality()).chi2LocalPosition;
      muontrkink[nmuons] = (muon1->combinedQuality()).trkKink;
      muonsegcom[nmuons] = int(1000. * muon::segmentCompatibility(*muon1)) / 1000.;
      
      muonpfiso[nmuons] = (muon1->pfIsolationR04().sumChargedHadronPt + max(0., muon1->pfIsolationR04().sumNeutralHadronEt + muon1->pfIsolationR04().sumPhotonEt - 0.5*muon1->pfIsolationR04().sumPUPt))/muon1->pt();
      
      muonisGoodGL[nmuons] = muon1->isGlobalMuon() && muon1->globalTrack()->normalizedChi2() < 3 && muon1->combinedQuality().chi2LocalPosition < 12 && muon1->combinedQuality().trkKink < 20; 
      muonisMed[nmuons] = (muon::isLooseMuon(*muon1))&&(muon1->innerTrack()->validFraction() > 0.8)&&(muonsegcom[nmuons] > (muonisGoodGL[nmuons] ? 0.303 : 0.451));
      muonisLoose[nmuons] = (muon::isLooseMuon(*muon1));
      
      muonpt[nmuons] = muon1->pt();
      muone[nmuons] = muon1->energy();
      muonp[nmuons] = trktrk->charge()*trktrk->p();
      muoneta[nmuons] = muon1->eta();
      muonphi[nmuons] = muon1->phi();
      
      muondrbm[nmuons] = trktrk->dxy(beamPoint);
  //  muontrkvtx[nmuons] = muon1->muonBestTrack()->dxy(beamPoint);
  //  muondz[nmuons] = muon1->muonBestTrack()->dz(beamPoint);
      muontrkvtx[nmuons] = muon1->muonBestTrack()->dxy(vertex.position());
      muondz[nmuons] = muon1->muonBestTrack()->dz(vertex.position());
      
      muonpter[nmuons] = trktrk->ptError();
      
      muonchi[nmuons] = trkglb->normalizedChi2();
      muonndf[nmuons] = (int)trkglb->ndof();
      
      muonhit[nmuons] = trkglb->hitPattern().numberOfValidMuonHits();
      muonmst[nmuons] = muon1->numberOfMatchedStations();
      muonpixhit[nmuons] = trktrk->hitPattern().numberOfValidPixelHits();
      muontrklay[nmuons] = trktrk->hitPattern().trackerLayersWithMeasurement();
      muonvalfrac[nmuons] = trktrk->validFraction();
      
      if (++nmuons>=njetmx) break;
    }
  }


  nelecs = 0;
  int iE1 = 0;
  for(const auto& electron1 : iEvent.get(tok_electrons_) ) {
    
    bool isPassMVAiso90 = electron1.electronID("mvaEleID-Fall17-iso-V2-wp90");
    bool isPassMVAnoiso90 = electron1.electronID("mvaEleID-Fall17-noIso-V2-wp90");
    elmvaid[nelecs] = isPassMVAiso90;                                                                          
    elmvaid_noIso[nelecs] = isPassMVAnoiso90;
   
    HepLorentzVector tmpelectron1(electron1.px(),electron1.py(),electron1.pz(), sqrt(electron1.p()*electron1.p()+el_mass*el_mass));
    iE1++;
    if (tmpelectron1.perp()<5.0) continue;
    
    elpt[nelecs] = electron1.charge()*electron1.pt();
    eleta[nelecs] = electron1.eta();
    elphi[nelecs] = electron1.phi();
    ele[nelecs] = electron1.ecalEnergy();
    elp[nelecs] = electron1.trackMomentumAtVtx().R();
    GsfTrackRef gsftrk1 = electron1.gsfTrack();
    eldxy[nelecs] = gsftrk1->dxy(beamPoint);
    eldz[nelecs] = gsftrk1->dz();

    if(++nelecs>=njetmx) break;
  }
  

  
  nphotons = 0;
  edm::Handle<edm::View<pat::Photon>> photons;

  edm::Handle <edm::ValueMap <bool> > mvaPhoIDSpring16GeneralPurposeV1wp90_reco;
  iEvent.getByToken(tok_mvaPhoIDSpring16GeneralPurposeV1wp90_reco, mvaPhoIDSpring16GeneralPurposeV1wp90_reco);

 
  iEvent.getByToken(tok_photons_, photons);
  if (photons.isValid()) {
	edm::View<pat::Photon>::const_iterator gamma1;
	int iPh1 = 0;
    for( gamma1 = photons->begin(); gamma1 != photons->end(); gamma1++ ) {
      if (!gamma1->isPhoton()) continue; 

        edm::Ptr<pat::Photon> pho_ptr(photons,iPh1);
	    phomvaid[nphotons] = (*mvaPhoIDSpring16GeneralPurposeV1wp90_reco)[pho_ptr];

		iPh1++;

		phoe[nphotons] = gamma1->energy();
		phoeta[nphotons] = gamma1->eta();
		phophi[nphotons] = gamma1->phi();
		phoe1by9[nphotons] = gamma1->maxEnergyXtal()/max(float(1),gamma1->e3x3());
		if (gamma1->hasConversionTracks()) { phoe1by9[nphotons] *= -1; }
		phoe9by25[nphotons] = gamma1->r9();
		phohadbyem[nphotons] = gamma1->hadronicOverEm();
		
		photrkiso[nphotons] = gamma1->trkSumPtSolidConeDR04();
		phoemiso[nphotons] = gamma1->ecalRecHitSumEtConeDR04();
		phohadiso[nphotons] = gamma1->hcalTowerSumEtConeDR04();
		phophoiso[nphotons] = gamma1->photonIso() ;
		phochhadiso[nphotons] = gamma1->chargedHadronIso();
		phoneuhadiso[nphotons] = gamma1->neutralHadronIso();
		phoietaieta[nphotons] = gamma1->sigmaIetaIeta();
		if (++nphotons>=njetmx) break;
	}
  }
  
   for(int jk=0; jk<nHLTmx; jk++) {
	  
	  switch(jk) {
		
	  case 0 :
	  ihlt01 = ihlttrg[jk];
	  prescl01 = iprescale[jk];
	  break;
	  
	  case 1 :
	  ihlt02 = ihlttrg[jk];
	  prescl02 = iprescale[jk];
	  break;
	  
	  case 2 :
	  ihlt03 = ihlttrg[jk];
	  prescl03 = iprescale[jk];
	  break;
	  
	}
  }	  
//  cout<<"done!"<<endl;
   T1->Fill();
}


// ------------ method called once each job just before starting event loop  ------------
void 
ExJets::beginJob()
{
	
  Nevt = 0;
/* 
  for(int ij=0; ij<nomassbins; ij++){
  massbins[ij] = 10*ij ;
  }

  rhobins[0] = 0.005;

  for(int ij=1; ij<norhobins; ij++){
    rhobins[ij] = width*rhobins[ij-1] ;
  }
*/
	////JEC /////
  L1FastAK4       = new JetCorrectorParameters(mJECL1FastFileAK4.c_str());
  L2RelativeAK4   = new JetCorrectorParameters(mJECL2RelativeFileAK4.c_str());
  L3AbsoluteAK4   = new JetCorrectorParameters(mJECL3AbsoluteFileAK4.c_str());
  L2L3ResidualAK4 = new JetCorrectorParameters(mJECL2L3ResidualFileAK4.c_str());

  vecL1FastAK4.push_back(*L1FastAK4);
  vecL2RelativeAK4.push_back(*L2RelativeAK4);
  vecL3AbsoluteAK4.push_back(*L3AbsoluteAK4);
  vecL2L3ResidualAK4.push_back(*L2L3ResidualAK4);
  
  jecL1FastAK4       = new FactorizedJetCorrector(vecL1FastAK4);
  jecL2RelativeAK4   = new FactorizedJetCorrector(vecL2RelativeAK4);
  jecL3AbsoluteAK4   = new FactorizedJetCorrector(vecL3AbsoluteAK4);
  jecL2L3ResidualAK4 = new FactorizedJetCorrector(vecL2L3ResidualAK4);
  
  L1FastAK8       = new JetCorrectorParameters(mJECL1FastFileAK8.c_str());
  L2RelativeAK8   = new JetCorrectorParameters(mJECL2RelativeFileAK8.c_str());
  L3AbsoluteAK8   = new JetCorrectorParameters(mJECL3AbsoluteFileAK8.c_str());
  L2L3ResidualAK8 = new JetCorrectorParameters(mJECL2L3ResidualFileAK8.c_str());

  vecL1FastAK8.push_back(*L1FastAK8);
  vecL2RelativeAK8.push_back(*L2RelativeAK8);
  vecL3AbsoluteAK8.push_back(*L3AbsoluteAK8);
  vecL2L3ResidualAK8.push_back(*L2L3ResidualAK8);
  
  jecL1FastAK8       = new FactorizedJetCorrector(vecL1FastAK8);
  jecL2RelativeAK8   = new FactorizedJetCorrector(vecL2RelativeAK8);
  jecL3AbsoluteAK8   = new FactorizedJetCorrector(vecL3AbsoluteAK8);
  jecL2L3ResidualAK8 = new FactorizedJetCorrector(vecL2L3ResidualAK8);


}

// ------------ method called once each job just after ending the event loop  ------------
void 
ExJets::endJob() 
{
// T1->Write();
 //fs->cd();
// fs->Write();
// fs->Close();

//delete fs;	
  
  theFile->cd();
  theFile->Write();
  theFile->Close();
}

// ------------ method called when starting to processes a run  ------------
void 
ExJets::beginRun(edm::Run const& iRun, edm::EventSetup const& pset)
{
bool changed(true);
hltPrescaleProvider_.init(iRun,pset,theHLTTag,changed);
HLTConfigProvider const&  hltConfig_ = hltPrescaleProvider_.hltConfigProvider();
hltConfig_.dump("Triggers");
hltConfig_.dump("PrescaleTable");
}

// ------------ method called when ending the processing of a run  ------------
void 
ExJets::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
ExJets::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
ExJets::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ExJets::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ExJets);
