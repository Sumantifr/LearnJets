ifeq ($(strip $(testExercise_JetsLearnJetsTP)),)
testExercise_JetsLearnJetsTP := self/src/Exercise_Jets/LearnJets/test
testExercise_JetsLearnJetsTP_files := $(patsubst src/Exercise_Jets/LearnJets/test/%,%,$(foreach file,test_catch2_*.cc,$(eval xfile:=$(wildcard src/Exercise_Jets/LearnJets/test/$(file)))$(if $(xfile),$(xfile),$(warning No such file exists: src/Exercise_Jets/LearnJets/test/$(file). Please fix src/Exercise_Jets/LearnJets/test/BuildFile.))))
testExercise_JetsLearnJetsTP_TEST_RUNNER_CMD :=  testExercise_JetsLearnJetsTP 
testExercise_JetsLearnJetsTP_BuildFile    := $(WORKINGDIR)/cache/bf/src/Exercise_Jets/LearnJets/test/BuildFile
testExercise_JetsLearnJetsTP_LOC_USE := self  FWCore/TestProcessor catch2
testExercise_JetsLearnJetsTP_PACKAGE := self/src/Exercise_Jets/LearnJets/test
ALL_PRODS += testExercise_JetsLearnJetsTP
testExercise_JetsLearnJetsTP_INIT_FUNC        += $$(eval $$(call Binary,testExercise_JetsLearnJetsTP,src/Exercise_Jets/LearnJets/test,src_Exercise_Jets_LearnJets_test,$(SCRAMSTORENAME_BIN),,$(SCRAMSTORENAME_TEST),test,$(SCRAMSTORENAME_LOGS)))
testExercise_JetsLearnJetsTP_CLASS := TEST
else
$(eval $(call MultipleWarningMsg,testExercise_JetsLearnJetsTP,src/Exercise_Jets/LearnJets/test))
endif
ALL_COMMONRULES += src_Exercise_Jets_LearnJets_test
src_Exercise_Jets_LearnJets_test_parent := Exercise_Jets/LearnJets
src_Exercise_Jets_LearnJets_test_INIT_FUNC += $$(eval $$(call CommonProductRules,src_Exercise_Jets_LearnJets_test,src/Exercise_Jets/LearnJets/test,TEST))
