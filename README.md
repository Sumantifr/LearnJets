# LearnJets
Go to work/private directory

(after logging into lxplus type "cd work/private")

First try cloning the repository and run the code using following commands:

git clone https://github.com/Sumantifr/LearnJets .

cd CMSSW_10_5_0/src/

scram b -j8

cd Exercise_Jets/LearnJets/test

cmsRun -n 0 JEC_MC_MINIAOD_cfg.py

If it doesn't work, the steps below.

Type the following commands in terminal

cmsrel CMSSW_10_5_0

cd CMSSW_10_5_0/src

mkdir Exercise_Jets

cd Exercise_Jets/

cmsenv

mkedanlzr LearnJets

cd LearnJets/plugins/

Now, copy NTuplizer_jets.cc from https://github.com/Sumantifr/LearnJets/tree/master/CMSSW_10_5_0/src/Exercise_Jets/LearnJets/plugins here

Paste everything in https://github.com/Sumantifr/LearnJets/tree/master/CMSSW_10_5_0/src/Exercise_Jets/LearnJets/plugins/BuildFile.xml  in the BuildFile.xml file, which is already here

Type following command to compile the code

scram b -j8

now, go to 'test' directory (cd ../test/)

copy everything from https://github.com/Sumantifr/LearnJets/tree/master/CMSSW_10_5_0/src/Exercise_Jets/LearnJets/test here

now run program using "cmsRun -n 0 JEC_MC_MINIAOD_cfg.py"
