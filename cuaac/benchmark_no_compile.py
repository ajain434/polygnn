import polygnn
import polygnn_trainer as pt
import pandas as pd
import torch
import time
# device = "cpu" # for GPU, use "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ###########################################################################
# Leave 'bond_config', 'atom_config' and 'featurization_scheme' as they are.
# ###########################################################################
bond_config = polygnn.featurize.BondConfig(True, True, True)
atom_config = polygnn.featurize.AtomConfig(
    True,
    True,
    True,
    True,
    True,
    True,
    combo_hybrid=False,
    aromatic=True,
)
featurization_scheme = "monocycle"
# ###########################################################################

root_dir = "/home/hice1/sshukla67/scratch/polygnn/trained_models/thermal" # you may need to change this, see below.

# 'data' is the data to predict. Two columns are required. The
# first called "smiles_string". This specifies which polymers you
# want to predict on. Both linear and ladder homopolymers are
# supported. The second column is called "prop". This specifies
# which properties you want to be predicted. The valid values of
# "prop" depend on the choice of "root_dir". Look at the README
# for lists of valid choices.
# data = pd.DataFrame(
#     {"smiles_string": ["*CC*", "*CO*"], "prop": ["exp_Tg__K", "exp_Tg__K"]},
# ) # change the values in this data frame to suit your use case.
# data = pd.read_csv('validation_tg.csv')
data = pd.read_parquet('cuaac_bert_gnn_pred.parquet')
data = data.head(10)
data = data[['smiles_string']]
data = pd.concat([data, data])
data['prop'] = 'exp_Tg__K'

start_time_1 = time.time()

# data.rename(columns={'product':'smiles_string'}, inplace=True)
# data = data.head(500000)
# The rest of the code does not need to be changed.
scalers = pt.load2.load_scalers(root_dir)
ensemble = pt.load.load_ensemble(
    root_dir,
    polygnn.models.polyGNN,
    device,
    {
        "node_size": atom_config.n_features,
        "edge_size": bond_config.n_features,
        "selector_dim": len(scalers),
    },
)
smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
    x,
    bond_config,
    atom_config,
    featurization_scheme,
)
end_time_1 = time.time()

print("Time till inference",end_time_1 - start_time_1)

start_time_2 = time.time()


y, y_mean_hat, y_std_hat, _selectors = pt.infer.eval_ensemble(
    model=ensemble,
    root_dir=root_dir,
    dataframe=data,
    smiles_featurizer=smiles_featurizer,
    device=device,
    ensemble_kwargs_dict={"monte_carlo": False},
)
end_time_2 = time.time()

print("Time for inference",end_time_2 - start_time_2)

print(y_mean_hat, y_std_hat)
data["Tg_pred"] = y_mean_hat
data["Tg_pred_std"] = y_std_hat
data_Tg = data