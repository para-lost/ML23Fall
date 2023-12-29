import synapseclient 
import synapseutils 

syn = synapseclient.Synapse() 
syn.login('jiaxinge','12345678syn') 
files = synapseutils.syncFromSynapse(syn, 'syn3193805') 
for file_info in files:
    print(file_info['path'])