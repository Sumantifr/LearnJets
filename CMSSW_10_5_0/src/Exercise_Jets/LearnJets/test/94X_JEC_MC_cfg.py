# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import sys

from RecoJets.Configuration.RecoPFJets_cff import *
from RecoJets.Configuration.RecoGenJets_cff import ak4GenJets, ak8GenJets
from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters
from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff import *
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.patSequences_cff import *
from PhysicsTools.PatAlgos.patTemplate_cfg import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff import *

## Modified version of jetToolBox from https://github.com/cms-jet/jetToolbox
## Options for PUMethod: Puppi, CS, SK, CHS

# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

process = cms.Process("Combined")
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Conditions
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('RecoJets.Configuration.GenJetParticles_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')
process.load('RecoJets.JetProducers.TrackJetParameters_cfi')
process.load('RecoJets.JetProducers.PileupJetIDParams_cfi')

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

from RecoJets.Configuration.GenJetParticles_cff import *

#process.GlobalTag.globaltag = "102X_mc2017_realistic_v6"
process.GlobalTag.globaltag = "94X_mc2017_realistic_v14"#"80X_mcRun2_asymptotic_v14"#"90X_upgrade2023_realistic_v1"  #"80X_mcRun2_asymptotic_2016_TrancheIV_v8" #"80X_mcRun2_asymptotic_2016_miniAODv2_v0"
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

from PhysicsTools.PatAlgos.tools.coreTools import *
process.load("PhysicsTools.PatAlgos.patSequences_cff")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Input
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

inFiles = cms.untracked.vstring(
'root://cmsxrootd.fnal.gov//store/mc/RunIIFall17MiniAODv2/TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/60000/FE8BE7BC-FDB6-E811-8B3E-001E675045FD.root'
   )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
#process.firstEvent = cms.untracked.PSet(input = cms.untracked.int32(5000))
process.source = cms.Source("PoolSource", fileNames = inFiles )


from jetToolbox_cff import jetToolbox

process.p = cms.Path()

#jetToolbox( process, 'ak8', 'ak8JetSubs', 'out', addSoftDropSubjets=True, addNsub=True, PUMethod='Puppi', miniAOD=True, runOnMC=True, addSoftDrop=True, JETCorrPayload= 'None')
jetToolbox( process, 'ak8', 'ak8JetSubs', 'out', PUMethod='CHS', miniAOD=True, runOnMC=False)
#jetToolbox( process, 'ak8', 'ak8JetSubs', 'out', PUMethod='Puppi', miniAOD=True, Cut = 'pt>170.', addSoftDrop=True,  betaCut=1.0,  zCutSD=0.1, addSoftDropSubjets=True, runOnMC=True)
##jetToolbox( process, 'ak8', 'ak8JetSubs', 'out', PUMethod='Puppi', addSoftDrop=True,  betaCut=0,  zCutSD=0.05, addSoftDropSubjets=True, runOnMC=True, JETCorrPayload= 'None')
#jetToolbox( process, 'ak8', 'jetSequence', 'out', PUMethod='CHS', miniAOD=True, runOnMC=True, JETCorrPayload= 'None')
#jetToolbox( process, 'ak4', 'jetSequence', 'out', PUMethod='CHS', miniAOD=True, runOnMC=True, JETCorrPayload= 'None')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Services
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.load('CommonTools.UtilAlgos.TFileService_cfi')
##process.TFileService.fileName=cms.string('DATA_ProcessedTreeProducer_2.root')

process.TFileService = cms.Service("TFileService",
fileName = cms.string('hist_jerc_l5.root')             #largest data till April5,2016 
)

process.patJets.addTagInfos = True

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
# Electron IDs for AOD/MINIAOD
switchOnVIDElectronIdProducer(process, DataFormat.MiniAOD)

# define which IDs to produce
el_id_modules = [
##    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff"
    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff",
    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff"
#    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff",
#    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff"
##    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff"
]
# Add them to the VID producer
for iModule in el_id_modules:

	setupAllVIDIdsInModule(process, iModule, setupVIDElectronSelection)

switchOnVIDPhotonIdProducer(process, DataFormat.MiniAOD)

pho_id_modules = [
	"RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff",
	"RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff"
]

for iModule in pho_id_modules:

	setupAllVIDIdsInModule(process, iModule, setupVIDPhotonSelection)

from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.Configuration.RecoGenJets_cff")

process.ak4GenJetsSoftDrop = process.ak4GenJets.clone(
    src = cms.InputTag("genParticlesForJetsNoNu"),
    useSoftDrop = cms.bool(True),
    zcut = cms.double(0.1),
    beta = cms.double(0.0),
    R0   = cms.double(0.4),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )
process.mcjets =  cms.EDAnalyzer('ExJets',

         Data =  cms.untracked.bool(False),
         MonteCarlo =  cms.untracked.bool(True),
         isReco = cms.untracked.bool(True),
         ReRECO = cms.untracked.bool(True),
         SoftDrop_ON =  cms.untracked.bool(True),
         RootFileName = cms.untracked.string('rootuple_jerc_l5.root'),

         softdropmass  = cms.untracked.string("ak8PFJetsSoftDropMass"),#ak8PFJetsPuppiSoftDropMass"),#('ak8PFJetsPuppiSoftDropMass'),

         minPt = cms.untracked.double(15.),
         maxEta = cms.untracked.double(3.),
         maxGenEta = cms.untracked.double(5.),
         AK8PtCut = cms.untracked.double(300.),
         nkTsub = cms.untracked.int32(2),

        Rsub_trim = cms.untracked.double(0.2),
        fcut_trim = cms.untracked.double(0.3),
        Rcut_prun = cms.untracked.double(0.2),
        zcut_prun = cms.untracked.double(0.1),
        beta = cms.untracked.double(0.),
        z_cut = cms.untracked.double(0.1),

        Beamspot = cms.InputTag("offlineBeamSpot"),
        PrimaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
        SecondaryVertices = cms.InputTag("slimmedSecondaryVertices"),
        slimmedAddPileupInfo = cms.InputTag("slimmedAddPileupInfo"),
        PFMet = cms.InputTag("slimmedMETs"),
        PFChMet  = cms.InputTag("slimmedMETs"),
        GENMet  = cms.InputTag("genMetTrue","","SIM"),
        Generator = cms.InputTag("generator"),
        HistWeight = cms.untracked.double(1.0),#0.53273),

        ## rho #######################################
        srcPFRho        = cms.InputTag('fixedGridRhoFastjetAll'),
        ## jec services ##############################

        PFRho = cms.InputTag("fixedGridRhoFastjetAll"),

         LHEEventProductInputTag = cms.InputTag('externalLHEProducer'),
         GenEventProductInputTag = cms.InputTag('generator'),

#        PFJetsAK8 = cms.InputTag("slimmedJetsAK8"),
#        PFJetsAK8 = cms.InputTag("selectedPatJetsAK8PFPuppiSoftDropPacked","SubJets","Combined"),
#        PFJetsAK8 = cms.InputTag("selectedPatJetsAK8PFCHS","","Combined"),
         PFJetsAK8 = cms.InputTag("selectedPatJetsAK8PFCHS"),
         PFJetsAK4 = cms.InputTag("slimmedJets"),
         GENJetAK8 = cms.InputTag("slimmedGenJetsAK8"),
         GENJetAK4 = cms.InputTag("slimmedGenJets"),#,"","PAT"),
         Muons = cms.InputTag("slimmedMuons"),#,"","PAT"),
         Electrons = cms.InputTag("slimmedElectrons"),#,"","PAT"),#("gsfElectrons"),
         Photons = cms.InputTag("slimmedPhotons"),#,"","PAT"),
         GenParticles = cms.InputTag("prunedGenParticles"),#("prunedGenParticles"),#("packedGenParticles"),

        #label_mvaEleIDSpring16GeneralPurposeV1wploose_noIso_reco = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp90"),
         #label_mvaEleIDSpring16GeneralPurposeV1wploose_reco = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp90"),
        label_mvaPhoIDSpring16GeneralPurposeV1wploose_reco = cms.InputTag("egmPhotonIDs:mvaPhoID-RunIIFall17-v1p1-wp90"),

         btag_CMVA_name = cms.untracked.string("pfCombinedMVAV2BJetTags"),
         btag_CSV_name = cms.untracked.string("pfCombinedInclusiveSecondaryVertexV2BJetTags"),

         HistFill = cms.untracked.bool(True),

         jecL1FastFileAK4          = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L1FastJet_AK4PFchs.txt'),
         jecL1FastFileAK8          = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L1FastJet_AK8PFPuppi.txt'),
         jecL2RelativeFileAK4      = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L2Relative_AK4PFchs.txt'),
         jecL2RelativeFileAK8      = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L2Relative_AK8PFPuppi.txt'),
         jecL3AbsoluteFileAK4      = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L3Absolute_AK4PFchs.txt'),
         jecL3AbsoluteFileAK8      = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L3Absolute_AK8PFPuppi.txt'),
         jecL2L3ResidualFileAK4    = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L2L3Residual_AK4PFchs.txt'),
         jecL2L3ResidualFileAK8    = cms.string('Autumn18_V19_MC/Autumn18_V19_MC_L2L3Residual_AK8PFPuppi.txt'),

         PtResoFileAK4  = cms.string('Autumn18_V7_MC_PtResolution_AK4PFchs.txt'),
         PtResoFileAK8  = cms.string('Autumn18_V7_MC_PtResolution_AK8PFPuppi.txt'),
         PtSFFileAK4 = cms.string('Autumn18_V7_MC_SF_AK4PFchs.txt'),
         PtSFFileAK8 = cms.string('Autumn18_V7_MC_SF_AK8PFPuppi.txt'),

         HBHENoiseFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResult"),
         HBHENoiseFilterResultNoMinZLabel = cms.InputTag("HBHENoiseFilterResultProducerNoMinZ", "HBHENoiseFilterResult"),

         JECUncFileAK4 = cms.string("Autumn18_V19_MC/Autumn18_V19_MC_UncertaintySources_AK4PFchs.txt"),
         JECUncFileAK8 = cms.string("Autumn18_V19_MC/Autumn18_V19_MC_UncertaintySources_AK8PFPuppi.txt"),

         bits = cms.InputTag("TriggerResults","","HLT"),
         prescales = cms.InputTag("patTrigger","","RECO"),
         objects = cms.InputTag("slimmedPatTrigger")
)


patJetsAK4 = process.updatedPatJetsAK4PFCHS
process.load('RecoJets.JetProducers.QGTagger_cfi')
patAlgosToolsTask.add(process.QGTagger)
process.QGTagger.srcJets=cms.InputTag("slimmedJets")
process.QGTagger.srcVertexCollection=cms.InputTag("offlineSlimmedPrimaryVertices")
patJetsAK4.userData.userFloats.src += ['QGTagger:qgLikelihood']
#process.out.outputCommands += ['keep *_QGTagger_*_*']

#process.ak4GenJetSD = cms.Path(process.ak4GenJetsSoftDrop)

#===== MET Filters ==

process.goodVertices = cms.EDFilter("VertexSelector",
   filter = cms.bool(False),
   src = cms.InputTag("offlineSlimmedPrimaryVertices"),
   cut = cms.string("!isFake && ndof >= 4 && abs(z) <= 24 && position.rho <= 2"),
)
process.load('RecoMET.METFilters.primaryVertexFilter_cfi')
process.primaryVertexFilter.vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices")
process.load('RecoMET.METFilters.globalTightHalo2016Filter_cfi')
process.load('RecoMET.METFilters.globalSuperTightHalo2016Filter_cfi')
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.load('CommonTools.RecoAlgos.HBHENoiseFilter_cfi')
process.HBHENoiseFilterResultProducerNoMinZ = process.HBHENoiseFilterResultProducer.clone(minZeros = cms.int32(99999))
process.load('RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi')

process.load('RecoMET.METFilters.BadPFMuonFilter_cfi')
process.BadPFMuonFilter.muons = cms.InputTag("slimmedMuons")
process.BadPFMuonFilter.PFCandidates = cms.InputTag("packedPFCandidates")

process.load('RecoMET.METFilters.BadChargedCandidateFilter_cfi')
process.BadChargedCandidateFilter.muons = cms.InputTag("slimmedMuons")
process.BadChargedCandidateFilter.PFCandidates = cms.InputTag("packedPFCandidates")

process.load('RecoMET.METFilters.eeBadScFilter_cfi')
process.eeBadScFilter.EERecHitSource = cms.InputTag('reducedEgamma','reducedEERecHits')

process.load('RecoMET.METFilters.ecalBadCalibFilter_cfi')

baddetEcallist = cms.vuint32(
    [872439604,872422825,872420274,872423218,
     872423215,872416066,872435036,872439336,
     872420273,872436907,872420147,872439731,
     872436657,872420397,872439732,872439339,
     872439603,872422436,872439861,872437051,
     872437052,872420649,872422436,872421950,
     872437185,872422564,872421566,872421695,
     872421955,872421567,872437184,872421951,
     872421694,872437056,872437057,872437313])

process.ecalBadCalibReducedMINIAODFilter = cms.EDFilter(
    "EcalBadCalibFilter",
    EcalRecHitSource = cms.InputTag("reducedEgamma:reducedEERecHits"),
    ecalMinEt        = cms.double(50.),
    baddetEcal    = baddetEcallist, 
    taggingMode = cms.bool(True),
    debug = cms.bool(False)
    )

process.allMetFilterPaths=cms.Sequence(process.primaryVertexFilter*process.globalSuperTightHalo2016Filter*process.HBHENoiseFilter*process.HBHENoiseIsoFilter*process.EcalDeadCellTriggerPrimitiveFilter*process.BadPFMuonFilter);#*process.BadChargedCandidateFilter)#*process.eeBadScFilter)

#process.allMetFilterPaths=cms.Sequence(process.primaryVertexFilter*process.HBHENoiseFilter*process.globalTightHalo2016Filter*process.EcalDeadCellTriggerPrimitiveFilter*process.BadPFMuonSummer16Filter*process.BadChargedCandidateSummer16Filter)#*process.eeBadScFilter)

process.p = cms.Path(process.egmGsfElectronIDSequence* 
		    process.egmPhotonIDSequence* 
		     process.HBHENoiseFilterResultProducer*process.HBHENoiseFilterResultProducerNoMinZ*
		     process.allMetFilterPaths*
		     process.ecalBadCalibReducedMINIAODFilter*
#		     process.egmGsfElectronIDSequence*
		     process.mcjets)
#process.p += process.mcjets

#Try scheduled processs
#process.path = cms.Path( process.mcjet.##			*process.ak5	)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )
process.options.allowUnscheduled = cms.untracked.bool(True)

#process.options.numberOfThreads=cms.untracked.uint32(2)
#process.options.numberOfStreams=cms.untracked.uint32(0)
