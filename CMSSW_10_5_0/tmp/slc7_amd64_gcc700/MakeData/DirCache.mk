ALL_SUBSYSTEMS+=Exercise_Jets
subdirs_src_Exercise_Jets = src_Exercise_Jets_LearnJets
ALL_PACKAGES += Exercise_Jets/LearnJets
subdirs_src_Exercise_Jets_LearnJets := src_Exercise_Jets_LearnJets_plugins src_Exercise_Jets_LearnJets_python src_Exercise_Jets_LearnJets_test
ifeq ($(strip $(PyExercise_JetsLearnJets)),)
PyExercise_JetsLearnJets := self/src/Exercise_Jets/LearnJets/python
src_Exercise_Jets_LearnJets_python_parent := 
ALL_PYTHON_DIRS += $(patsubst src/%,%,src/Exercise_Jets/LearnJets/python)
PyExercise_JetsLearnJets_files := $(patsubst src/Exercise_Jets/LearnJets/python/%,%,$(wildcard $(foreach dir,src/Exercise_Jets/LearnJets/python ,$(foreach ext,$(SRC_FILES_SUFFIXES),$(dir)/*.$(ext)))))
PyExercise_JetsLearnJets_LOC_USE := self  
PyExercise_JetsLearnJets_PACKAGE := self/src/Exercise_Jets/LearnJets/python
ALL_PRODS += PyExercise_JetsLearnJets
PyExercise_JetsLearnJets_INIT_FUNC        += $$(eval $$(call PythonProduct,PyExercise_JetsLearnJets,src/Exercise_Jets/LearnJets/python,src_Exercise_Jets_LearnJets_python,1,1,$(SCRAMSTORENAME_PYTHON),$(SCRAMSTORENAME_LIB),,))
else
$(eval $(call MultipleWarningMsg,PyExercise_JetsLearnJets,src/Exercise_Jets/LearnJets/python))
endif
ALL_COMMONRULES += src_Exercise_Jets_LearnJets_python
src_Exercise_Jets_LearnJets_python_INIT_FUNC += $$(eval $$(call CommonProductRules,src_Exercise_Jets_LearnJets_python,src/Exercise_Jets/LearnJets/python,PYTHON))
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
