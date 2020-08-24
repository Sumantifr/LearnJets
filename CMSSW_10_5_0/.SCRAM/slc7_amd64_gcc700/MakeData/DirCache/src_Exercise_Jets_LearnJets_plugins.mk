ifeq ($(strip $(Exercise_JetsLearnJetsAuto)),)
Exercise_JetsLearnJetsAuto := self/src/Exercise_Jets/LearnJets/plugins
PLUGINS:=yes
Exercise_JetsLearnJetsAuto_files := $(patsubst src/Exercise_Jets/LearnJets/plugins/%,%,$(wildcard $(foreach dir,src/Exercise_Jets/LearnJets/plugins ,$(foreach ext,$(SRC_FILES_SUFFIXES),$(dir)/*.$(ext)))))
Exercise_JetsLearnJetsAuto_BuildFile    := $(WORKINGDIR)/cache/bf/src/Exercise_Jets/LearnJets/plugins/BuildFile
Exercise_JetsLearnJetsAuto_LOC_USE := self  FWCore/Framework FWCore/PluginManager FWCore/ParameterSet DataFormats/ParticleFlowCandidate DataFormats/JetReco DataFormats/METReco CommonTools/UtilAlgos CommonTools/RecoAlgos DataFormats/CLHEP DataFormats/Common DataFormats/Candidate DataFormats/HLTReco JetMETCorrections/JetCorrector FWCore/Common DataFormats/HcalRecHit DataFormats/EgammaCandidates DataFormats/MuonReco DataFormats/VertexReco DataFormats/HcalDigi DataFormats/HcalDetId CalibFormats/HcalObjects DPGAnalysis/Skims GeneratorInterface/Pythia8Interface root JetMETCorrections/Modules DataFormats/PatCandidates CondFormats/JetMETObjects SimDataFormats/GeneratorProducts fastjet fastjet-contrib
Exercise_JetsLearnJetsAuto_PRE_INIT_FUNC += $$(eval $$(call edmPlugin,Exercise_JetsLearnJetsAuto,Exercise_JetsLearnJetsAuto,$(SCRAMSTORENAME_LIB),src/Exercise_Jets/LearnJets/plugins))
Exercise_JetsLearnJetsAuto_PACKAGE := self/src/Exercise_Jets/LearnJets/plugins
ALL_PRODS += Exercise_JetsLearnJetsAuto
Exercise_Jets/LearnJets_forbigobj+=Exercise_JetsLearnJetsAuto
Exercise_JetsLearnJetsAuto_INIT_FUNC        += $$(eval $$(call Library,Exercise_JetsLearnJetsAuto,src/Exercise_Jets/LearnJets/plugins,src_Exercise_Jets_LearnJets_plugins,$(SCRAMSTORENAME_BIN),,$(SCRAMSTORENAME_LIB),$(SCRAMSTORENAME_LOGS),edm))
Exercise_JetsLearnJetsAuto_CLASS := LIBRARY
else
$(eval $(call MultipleWarningMsg,Exercise_JetsLearnJetsAuto,src/Exercise_Jets/LearnJets/plugins))
endif
ALL_COMMONRULES += src_Exercise_Jets_LearnJets_plugins
src_Exercise_Jets_LearnJets_plugins_parent := Exercise_Jets/LearnJets
src_Exercise_Jets_LearnJets_plugins_INIT_FUNC += $$(eval $$(call CommonProductRules,src_Exercise_Jets_LearnJets_plugins,src/Exercise_Jets/LearnJets/plugins,PLUGINS))
