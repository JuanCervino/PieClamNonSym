import torch



import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')

from datasets.import_dataset import import_dataset
from trainer import Trainer
from utils.plotting import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




data_bipart = import_dataset('bipartiteDirected')
# data_bipart = import_dataset('bipartite')
config_triplets=[
# ['clamiter_init', 'dim_feat', 2],

]

trainer_ieclam_bipart = Trainer(dataset=data_bipart, 
                                 model_name='ieclam',
                                 device='cpu',
                                 config_triplets_to_change=config_triplets,
                                 use_global_config_base=False
                                )

_ = trainer_ieclam_bipart.train(plot_every=100000)
trainer_ieclam_bipart.plot_state(things_to_plot=['adj', '2dgraphs', 'feats'], draw_edges=True)