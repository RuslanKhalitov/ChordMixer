from .genbank import GenbankDataModule
from .longdoc import LongdocDataModule
from .adding  import AddingDataModule

from .lra_listops import ListOpsDataModule
from .lra_listops_var import ListOpsDataModuleVar
from .lra_image import CIFAR10
from .lra_text import IMDBDataset
from .lra_retrieval import AANDataModule
from .lra_pathfinder import Pathfinder
from .lra_pathfinderx import PathfinderX

def map_dataset(
    data_dir,
    taskname,
    num_workers,
    batch_size,
    diff_lengths
):
    if taskname == 'longdoc':
        return LongdocDataModule(
            data_dir=data_dir,
            taskname=taskname,
            num_workers=num_workers,
            batch_size=batch_size,
            diff_lengths=diff_lengths    
        )
    elif taskname in ['Carassius_Labeo', 'Danio_Cyprinus', 'Sus_Bos', 'Mus_Rattus']:
        print('TASKNAME', taskname)
        return GenbankDataModule(
            data_dir,
            taskname,
            num_workers,
            batch_size,
        )
    elif taskname in ['adding_200', 'adding_1000', 'adding_16000', 'adding_128000']:
        return AddingDataModule(
            data_dir,
            taskname,
            num_workers,
            batch_size
        )
    elif taskname == 'listops':
        return ListOpsDataModule(
            data_dir=data_dir,
            num_workers=num_workers,
            batch_size=batch_size
        )
    elif taskname == 'listops_var':
        return ListOpsDataModuleVar(
            data_dir=data_dir,
            num_workers=num_workers,
            batch_size=batch_size
        )
    elif taskname == 'retrieval':
        return AANDataModule(
            data_dir,
            num_workers,
            batch_size           
        )
    elif taskname == 'image':
        return CIFAR10(
            data_dir,
            num_workers,
            batch_size           
        )
    elif taskname == 'text':
        return IMDBDataset(
            data_dir,
            num_workers,
            batch_size           
        )
    elif taskname == 'pathfinder':
        return Pathfinder(
            data_dir,
            num_workers,
            batch_size           
        )    
    elif taskname == 'pathfinderx':
        return PathfinderX(
            data_dir,
            num_workers,
            batch_size                
        )