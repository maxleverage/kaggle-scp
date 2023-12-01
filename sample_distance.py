import os
import sys

import keras.backend as K
from tensorflow import keras
import tensorflow as tf

import math as m
import numpy as np
import pandas as pd
import warnings

# Load in training data
train_path = os.getcwd() + '/Data/de_train.parquet'
df_train = pd.read_parquet(train_path)

# Define rwrmse
def rwrmse_numpy(y_true, y_pred):
	return np.mean(np.mean((y_true - y_pred) ** 2, axis=1) ** 0.5)

# Define cosine similarity
def cosine_similarity(y_true, y_pred):
	return np.mean(-K.eval(keras.losses.cosine_similarity(y_true, y_pred, axis=1)))

# Define correlation
def correlation(y_true, y_pred):
	corr_vals = []
	for i in range(y_true.shape[0]):
		corr_vals.append(np.corrcoef(y_true[i, :], y_pred[i, :])[1, 0])
	return np.mean(corr_vals)

# Calculate distance on base y_train data matrix
# {'T cells CD8+', 'NK cells', 'B cells', 'Myeloid cells', 'T cells CD4+', 'T regulatory cells'}
# B cells and myeloid cells in submission
comb = {}
target_cells = ['B cells', 'Myeloid cells']
for elem in set(df_train.cell_type):
	if elem not in target_cells:
		comb[elem] = target_cells

# Find matching sm_names between both cell types
b_cell_sm = list(df_train[df_train["cell_type"] == "B cells"]["sm_name"])
m_cell_sm = list(df_train[df_train["cell_type"] == "Myeloid cells"]["sm_name"])

# Ground truths for B cells and myeloid cells
b_model = df_train[df_train["cell_type"] == "B cells"]
m_model = df_train[df_train["cell_type"] == "Myeloid cells"]

# Choose distance
dist_type = "cosine"

# B cell and myeloid cell models
key_list = list(comb.keys())
# Dist dictionary
cell_dist = {}
for cell in key_list:
	result = []
	# Calculate first for B cells
	cell_model = df_train[df_train["cell_type"] == cell]
	base_model = cell_model[cell_model["sm_name"].isin(b_cell_sm)]
	# Calculate test for B cell and Myeloid cells
	cell_model_sm = list(base_model["sm_name"])
	# Get B cell ground truth
	b_cell_model = b_model[b_model["sm_name"].isin(cell_model_sm)]
	# Convert base_model and b_model_cell into np matrices
	base_model = np.array(base_model)[:, 5:].astype('float32')
	b_cell_model = np.array(b_cell_model)[:, 5:].astype('float32')
	# Calculate Frobenius norm
	if dist_type == "fro":
		dist = np.linalg.norm(b_cell_model - base_model, ord='fro')
	elif dist_type == "rwrmse":
		dist = rwrmse_numpy(base_model, b_cell_model)
	elif dist_type == "cosine":
		dist = cosine_similarity(base_model, b_cell_model)
	elif dist_type == "correlation":
		dist = correlation(base_model, b_cell_model)
	result.append(dist)
	# Calculate for myeloid cells
	cell_model = df_train[df_train["cell_type"] == cell]
	base_model = cell_model[cell_model["sm_name"].isin(m_cell_sm)]
	# Get sm list for prediction model
	cell_model_sm = list(base_model["sm_name"])
	# Form ground truth for myeloid cells
	m_cell_model = m_model[m_model["sm_name"].isin(cell_model_sm)]
	# Convert base and m_cell_model into np matrices
	base_model = np.array(base_model)[:, 5:].astype('float32')
	m_cell_model = np.array(m_cell_model)[:, 5:].astype('float32')
	# Calculuate Frobenius norm
	if dist_type == "fro":
		dist = np.linalg.norm(m_cell_model - base_model, ord='fro')
	elif dist_type == "rwrmse":
		dist = rwrmse_numpy(base_model, m_cell_model)
	elif dist_type == "cosine":
		dist = cosine_similarity(base_model, m_cell_model)
	elif dist_type == "correlation":
		dist = correlation(base_model, m_cell_model)
	result.append(dist)
	# Assign to dictionary
	cell_dist[cell] = result

# Scale distances calculated
for cell in key_list:
	cell_dist[cell] = np.mean(cell_dist[cell]) / 1

# Calculuate by inverse log
base = 1.5
for cell in key_list:
	cell_dist[cell] = 1/m.log(base + cell_dist[cell], base)

# Calculate distance from 1.
for cell in key_list:
	cell_dist[cell] = 1 - cell_dist[cell]

# Cell type numbers
cell_type_num = {'B cells': 17., 'Myeloid cells': 17., 'T regulatory cells': 146., 'T cells CD8+': 142, 'NK cells': 146., 'T cells CD4+': 146.}
n_b_myeloid = 34
cell_factor = {'T regulatory cells': 146./n_b_myeloid, 'T cells CD8+': 142./n_b_myeloid, 'NK cells': 146./n_b_myeloid, 'T cells CD4+': 146./n_b_myeloid}

# Normalised weights (relative to NK cells) non B / myeloid cells
# weight ^ 3 / W['NK cells'] ^ 3
normalised_dist = {'T regulatory cells': 0.6516022331609318, 'T cells CD8+': 0.8793234204962851, 'NK cells': 1., 'T cells CD4+': 0.7614053465132592}

# Exponent base
base_exp = 0.025

# Scaled weight
scaling_factor = {'T regulatory cells': (1 + base_exp * normalised_dist['T regulatory cells']) ** cell_factor['T regulatory cells'],
 	'T cells CD8+': (1 + base_exp * normalised_dist['T cells CD8+']) ** cell_factor['T cells CD8+'], 
 	'NK cells': (1 + base_exp * normalised_dist['NK cells']) ** cell_factor['NK cells'], 
 	'T cells CD4+': (1 + base_exp * normalised_dist['T cells CD4+']) ** cell_factor['T cells CD4+']}

# Scaling factors used
# scaling_factor = {'T regulatory cells': 1.0718517636591396, 'T cells CD8+': 1.0950687072395096, 'NK cells': 1.1118585488618329, 'T cells CD4+': 1.0843393196219417}

# Try same framework on SVHT denoised matrix
y_train = np.array(df_train.iloc[:, 5:])

# SVD
U, S, VT = np.linalg.svd(y_train, full_matrices=False)

# Calculate aspect ratio and cutoff
Beta = y_train.shape[0] / y_train.shape[1]
# Approximate w(B)
w_B = 0.56 * Beta ** 3 - 0.95 * Beta ** 2 + 1.82 * Beta + 1.43
# Higher w_B, higher tau threshold, lower number of columns kept
w_B_low = w_B - 0.02
w_B_high = w_B + 0.02
med_S = np.median(S)
tau = w_B_high * med_S

# Optimal modes
# Can consider dropping a few modes
q = np.max(np.where(S > tau))
U, S, VT = U[:, :(q+1)], np.diag(S[:(q+1)]), VT[:(q+1), :]

# Calculate denoised y_train
y_train = U @ S @ VT

# Assign y_train to df
df_train.iloc[:, 5:] = y_train

# Perform same distance metric
comb = {}
target_cells = ['B cells', 'Myeloid cells']
for elem in set(df_train.cell_type):
	if elem not in target_cells:
		comb[elem] = target_cells

# Find matching sm_names between both cell types
b_cell_sm = list(df_train[df_train["cell_type"] == "B cells"]["sm_name"])
m_cell_sm = list(df_train[df_train["cell_type"] == "Myeloid cells"]["sm_name"])

# Ground truths for B cells and myeloid cells
b_model = df_train[df_train["cell_type"] == "B cells"]
m_model = df_train[df_train["cell_type"] == "Myeloid cells"]

# Get Frboenius norms of models on ground truth
key_list = list(comb.keys())
# Dist dictionary
cell_dist = {}
for cell in key_list:
	result = []
	# Calculate first for B cells
	cell_model = df_train[df_train["cell_type"] == cell]
	base_model = cell_model[cell_model["sm_name"].isin(b_cell_sm)]
	# Calculate test for B cell and Myeloid cells
	cell_model_sm = list(base_model["sm_name"])
	# Get B cell ground truth
	b_cell_model = b_model[b_model["sm_name"].isin(cell_model_sm)]
	# Convert base_model and b_model_cell into np matrices
	base_model = np.array(base_model)[:, 5:]
	b_cell_model = np.array(b_cell_model)[:, 5:]
	# Calculate Frobenius norm
	dist = np.linalg.norm(b_cell_model - base_model, ord='fro')
	result.append(dist)
	# Calculate for myeloid cells
	cell_model = df_train[df_train["cell_type"] == cell]
	base_model = cell_model[cell_model["sm_name"].isin(m_cell_sm)]
	# Get sm list for prediction model
	cell_model_sm = list(base_model["sm_name"])
	# Form ground truth for myeloid cells
	m_cell_model = m_model[m_model["sm_name"].isin(cell_model_sm)]
	# Convert base and m_cell_model into np matrices
	base_model = np.array(base_model)[:, 5:]
	m_cell_model = np.array(m_cell_model)[:, 5:]
	# Calculuate Frobenius norm
	dist = np.linalg.norm(m_cell_model - base_model, ord='fro')
	result.append(dist)
	# Assign to dictionary
	cell_dist[cell] = result

# Scale distances calculated
for cell in key_list:
	cell_dist[cell] = np.mean(cell_dist[cell]) / 1e4

# Calculuate by inverse log
base = 1.5
for cell in key_list:
	cell_dist[cell] = 1/m.log(base + cell_dist[cell], base)

# Output result
# {'T regulatory cells': 0.6527382717652584, 'T cells CD8+': 0.7218603643498793, 'NK cells': 0.7543927723023031, 'T cells CD4+': 0.6878647561303267}

# Perform same metric on SVHT determined sub components
y_embed = pd.DataFrame(U @ S)

# Get dataframe without gene expressions
df_train_no_gene = df_train.iloc[:, :5]
df_train = df_train_no_gene.join(y_embed)

# Get combinations of cells
comb = {}
target_cells = ['B cells', 'Myeloid cells']
for elem in set(df_train.cell_type):
	if elem not in target_cells:
		comb[elem] = target_cells

# Find matching sm_names between both cell types
b_cell_sm = list(df_train[df_train["cell_type"] == "B cells"]["sm_name"])
m_cell_sm = list(df_train[df_train["cell_type"] == "Myeloid cells"]["sm_name"])

# Ground truths for B cells and myeloid cells
b_model = df_train[df_train["cell_type"] == "B cells"]
m_model = df_train[df_train["cell_type"] == "Myeloid cells"]

# Get Frboenius norms of models on ground truth
key_list = list(comb.keys())
# Dist dictionary
cell_dist = {}
for cell in key_list:
	result = []
	# Calculate first for B cells
	cell_model = df_train[df_train["cell_type"] == cell]
	base_model = cell_model[cell_model["sm_name"].isin(b_cell_sm)]
	# Calculate test for B cell and Myeloid cells
	cell_model_sm = list(base_model["sm_name"])
	# Get B cell ground truth
	b_cell_model = b_model[b_model["sm_name"].isin(cell_model_sm)]
	# Convert base_model and b_model_cell into np matrices
	base_model = np.array(base_model)[:, 5:]
	b_cell_model = np.array(b_cell_model)[:, 5:]
	# Calculate Frobenius norm
	dist = np.linalg.norm(b_cell_model - base_model, ord='fro')
	result.append(dist)
	# Calculate for myeloid cells
	cell_model = df_train[df_train["cell_type"] == cell]
	base_model = cell_model[cell_model["sm_name"].isin(m_cell_sm)]
	# Get sm list for prediction model
	cell_model_sm = list(base_model["sm_name"])
	# Form ground truth for myeloid cells
	m_cell_model = m_model[m_model["sm_name"].isin(cell_model_sm)]
	# Convert base and m_cell_model into np matrices
	base_model = np.array(base_model)[:, 5:]
	m_cell_model = np.array(m_cell_model)[:, 5:]
	# Calculuate Frobenius norm
	dist = np.linalg.norm(m_cell_model - base_model, ord='fro')
	result.append(dist)
	# Assign to dictionary
	cell_dist[cell] = result

# Scale distances calculated
for cell in key_list:
	cell_dist[cell] = np.mean(cell_dist[cell]) / 1e4

# Calculuate by inverse log
base = 1.5
for cell in key_list:
	cell_dist[cell] = 1/m.log(base + cell_dist[cell], base)

# Output result
# {'T regulatory cells': 0.6527382717652572, 'T cells CD8+': 0.7218603643498789, 'NK cells': 0.7543927723023031, 'T cells CD4+': 0.6878647561303274}

