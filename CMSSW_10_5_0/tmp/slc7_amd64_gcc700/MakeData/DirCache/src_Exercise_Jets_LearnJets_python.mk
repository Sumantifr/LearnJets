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
