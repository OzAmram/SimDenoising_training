import sys
import os
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))
from magiconfig import MagiConfig
import glob

energy = 850
num_events = '*'
num_files = 88
#num_files = 4
val_frac = 0.25
#currently only works with multiple files ... ?

def get_files(filetype):
    globstr = ""
    globstr = '/storage/local/data1/gpuscratch/oamram/denoise/May13Prod/ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'_part*.root'
    #if filetype == 'Fuzz': globstr = '/storage/local/data1/gpuscratch/oamram/denoise/May13Prod/ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'_part*.root'
    #elif filetype == 'Sharp': globstr = '/storage/local/data1/gpuscratch/oamram/denoise/May13Prod/ntup_photon_energy'+str(energy)+'*phi0.0_n'+str(num_events)+'_part*.root'
    #if filetype == 'Fuzz': globstr = 'root://cmseosmgm01.fnal.gov:1094//store/user/oamram/SimDenoise/ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'_part*.root'
    #elif filetype == 'Sharp': globstr = 'root://cmseosmgm01.fnal.gov:1094//store/user/oamram/SimDenoise/ntup_photon_energy'+str(energy)+'*phi0.0_n'+str(num_events)+'_part*.root'
    list_of_names = []
    for name in glob.glob(globstr):
        if(filetype == 'Sharp'): name = name.replace("ProductionCut10.0_", "")
        list_of_names.append(name)
    print("Found %i files" % len(list_of_names))
    list_of_names.sort()
    tot_num_files = min(num_files, len(list_of_names))
    num_train_files = int(round(tot_num_files * (1. - val_frac)))
    print("%i files for training, %i for val" % (num_train_files, tot_num_files - num_train_files))
    list_of_names_t = list_of_names[:num_train_files]
    list_of_names_v = list_of_names[num_train_files:tot_num_files]
    print(len(list_of_names_t), len(list_of_names_v))
    return list_of_names_t, list_of_names_v

fuzzy_t_files, fuzzy_v_files = get_files('Fuzz')
sharp_t_files, sharp_v_files = get_files('Sharp')

print(fuzzy_t_files)
print(fuzzy_v_files)

config = MagiConfig()
config.outf = 'out-train-june15-SetFeats500-largeTrain'
config.imageOnly = False
config.extraImages = False
config.applyAugs = False
config.numSetFeats = 4
config.batchSize = 50
config.epochs = 100
#config.epochs = 10
config.features = 100
#config.features = 64
config.kernelSize = 3
config.lr = 0.001
config.num_layers = 9
config.num_workers = 8
config.transform = ['sqrt', 'normalize']
config.patchSize = 100
config.trainfileFuzz = fuzzy_t_files
config.trainfileSharp = sharp_t_files
config.valfileFuzz = fuzzy_v_files
config.valfileSharp = sharp_v_files
config.numpy = os.path.join(config.outf, "out.npz")
config.model = os.path.join(config.outf, "net.pth")
config.testfileSharp = open('test_files_sharp.txt').read().splitlines()
config.testfileFuzz = open('test_files_fuzz.txt').read().splitlines()
