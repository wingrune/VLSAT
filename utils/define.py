#ROOT_PATH = '/home/wingrune/3rscan-datasets/3DSSG/data/3RScan/'
#DATA_PATH = '/home/wingrune/3rscan-datasets/3DSSG/data/3RScan/data/3RScan/'
ROOT_PATH = '/home/wingrune/3rscan-datasets/ScanNet' 
DATA_PATH = '/home/wingrune/3rscan-datasets/ScanNet/scans'
SCANNET_DATA_PATH = '/home/wingrune/3rscan-datasets/ScanNet/scans'
SCANNET_SPLIT_TRAIN = '/home/wingrune/3rscan-datasets/ScanNet/split/scannet_sample.txt'
SCANNET_SPLIT_VAL = '/path/to/scannet/Tasks/Benchmark/scannetv2_val.txt'
PATH_LABEL_MAPPING = '/home/wingrune/3rscan-datasets/3DSSG/data/3RScan/data/3RScan/3RScan.v2 Semantic Classes - Mapping.csv'
PATH_CLASS160_FILE = '/home/wingrune/3rscan-datasets/CVPR2023-VLSAT/data/3DSSG_subset/classes.txt'

FILE_PATH = ROOT_PATH+'files/'
Scan3RJson_PATH = FILE_PATH+'3RScan.json'
LABEL_MAPPING_FILE = FILE_PATH+'3RScan.v2 Semantic Classes - Mapping.csv'
CLASS160_FILE = FILE_PATH+'classes160.txt'

# 3RScan file names
LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.align.annotated.v2.ply'
SEMSEG_FILE_NAME = 'semseg.v2.json'
MTL_NAME = 'mesh.refined.mtl'
OBJ_NAME = 'mesh.refined.v2.obj'
TEXTURE_NAME = 'mesh.refined_0.png'

# ScanNet file names
SCANNET_SEG_SUBFIX = '_vh_clean_2.0.010000.segs.json'
SCANNET_AGGRE_SUBFIX = '.aggregation.json'
SCANNET_PLY_SUBFIX = '_vh_clean_2.labels.ply'


NAME_SAME_PART = 'same part'
