import sys
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))
from magiconfig import MagiConfig
import glob

energy = 850
num_events = '*'
num_files = 28
val_frac = 0.25
#currently only works with multiple files ... ?

def get_files(filetype):
    globstr = ""
    if filetype == 'Fuzz': globstr = '/storage/local/data1/gpuscratch/oamram/denoise/May13Prod/ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'_part*.root'
    elif filetype == 'Sharp': globstr = '/storage/local/data1/gpuscratch/oamram/denoise/May13Prod/ntup_photon_energy'+str(energy)+'*phi0.0_n'+str(num_events)+'_part*.root'
    #if filetype == 'Fuzz': globstr = '/storage/local/data1/gpuscratch/oamram/denoise/50by50/ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'_part*.root'
    #elif filetype == 'Sharp': globstr = '/storage/local/data1/gpuscratch/oamram/denoise/50by50/ntup_photon_energy'+str(energy)+'*phi0.0_n'+str(num_events)+'_part*.root'
    #if filetype == 'Fuzz': globstr = 'root://cmseosmgm01.fnal.gov:1094//store/user/oamram/SimDenoise/ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'_part*.root'
    #elif filetype == 'Sharp': globstr = 'root://cmseosmgm01.fnal.gov:1094//store/user/oamram/SimDenoise/ntup_photon_energy'+str(energy)+'*phi0.0_n'+str(num_events)+'_part*.root'
    list_of_names = []
    for name in glob.glob(globstr):
        list_of_names.append(name)
    list_of_names.sort()
    num_train_files = int(round(num_files * (1. - val_frac)))
    print("%i files for training, %i for val" % (num_train_files, num_files - num_train_files))
    list_of_names_t = list_of_names[:num_train_files]
    list_of_names_v = list_of_names[num_train_files:num_files]
    return list_of_names_t, list_of_names_v

fuzzy_t_files, fuzzy_v_files = get_files('Fuzz')
sharp_t_files, sharp_v_files = get_files('Sharp')

config = MagiConfig()
config.imageOnly = False
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
config.outf = 'out-train-may13-PointCloudCNN-4feats-NoAugs'
config.transform = ['sqrt', 'normalize']
config.patchSize = 100
config.trainfileFuzz = fuzzy_t_files
config.trainfileSharp = sharp_t_files
config.valfileFuzz = fuzzy_v_files
config.valfileSharp = sharp_v_files
