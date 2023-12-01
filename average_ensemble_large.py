import numpy as np 
import pandas as pd
import os
import hashlib

working_directory = os.getcwd() + '/Output'
os.chdir(working_directory)

# 80 models
models_list = ['ecfp_avg_ensemble_v126_c3b4eea3a8dc3aa113d1a6a.csv',
	'ecfp_avg_ensemble_v195_2fc65c100d52617c1b10935.csv',
	'ecfp_avg_ensemble_v196_e3817ec3de6b62652d28881.csv',
	'ecfp_avg_ensemble_v245_ddf5d728bb35e0dd16a8aff.csv',
	'ecfp_avg_ensemble_v435_068abfdb6adfe8c4e0c6fdc.csv',
	'ecfp_avg_ensemble_v422_afe85170ed936812cf85ee8.csv',
	'ecfp_avg_ensemble_v433_ff85d77be0e9d20a4f6563e.csv',
	'ecfp_avg_ensemble_v436_7c7a8979cb694d41d175fea.csv',
	'ecfp_avg_ensemble_v443_112e91e9cad4a3d8a18a863.csv',
	'ecfp_reg_8lselu_v100_171830d886f43773dcdcd9b.csv',
	'ecfp_regression_8lselu_5128_logPlogMR_1500e_16b_sscaled_svht_yembed_adamW_wB_high_weighted_mae_v95.csv',
	'ecfp_avg_ensemble_v423_c1b687288e19825d0418484.csv',
	'ecfp_reg_8lselu_stack_v159_1bfc6b8205e7ecc2c0e49c1.csv',
	'ecfp_reg_8lselu_multi_v277_f2b3c10f83149d05e14e2fc.csv',
	'ecfp_regression_8lselu_5128_logPlogMR_1000e_16b_sscaled_svht_yembed_adamW_wB_high_weighted_mae_v91.csv',
	'ecfp_reg_8lselu_stack_JTT_v215_1217ebef8fb08119034c642.csv',
	'ecfp_reg_8lselu_stack_JTT_v235_3012209a70888ebcafcc8d3.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v272_40c615c20dd047d8b92234f.csv',
	'ecfp_reg_8lselu_v122_c64ce82baf1db648e52f890.csv',
	'ecfp_reg_8lselu_stack_v139_3f7599b75f97f08597d86ee.csv',
	'ecfp_reg_8lselu_stack_v191_528d29ec39a91c3013146bb.csv',
	'ecfp_avg_ensemble_v246_4637c7f97913e30cafcfc0e.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v286_d56dae50677c9a480084edf.csv',
	'ecfp_reg_8lselu_multi_v296_6d5b3f67290502f18ed9bf0.csv',
	'ecfp_reg_8lselu_multi_v320_e7466da658f470f45f728e7.csv',
	'ecfp_reg_8lselu_multi_v342_b2d78d782f85443f796a74e.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v411_1afc99666e3253748f81464.csv',
	'ecfp_regression_8lselu_5128_logPlogMR_700e_16b_sscaled_svht_yembed_adamW_wB_high_weighted_mae_v89.csv',
	'ecfp_reg_8lselu_stack_JTT_v169_cdd7e2b3537df62bbd986ba.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v396_eadd28292a07db6838644eb.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v428_3f3aa1f81f75f1c64925116.csv',
	'ecfp_reg_8lselu_v111_162441495e2fec4d676ff17.csv',
	'ecfp_reg_8lselu_stack_JTT_v230_4f2fc085366ee519b73a0c8.csv',
	'ecfp_reg_8lselu_stack_JTT_v249_d9d6bb3a6baf362df9a2313.csv',
	'ecfp_reg_8lselu_multi_v287_837a6077115eb5fcec593ac.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v288_94830e026183110f55be251.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v409_71546644a0fd81fb196a225.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v414_33b5946d8f3a7f2b9567c1f.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v416_b773f74dd470a255a850aa4.csv',
	'ecfp_reg_8lselu_v114_fae11891d65e3614109558e.csv',
	'ecfp_reg_8lselu_stack_JTT_v163_d63b383244a186ac5502efa.csv',
	'ecfp_reg_8lselu_stack_v197_40e7f53b5107432b9431893.csv',
	'ecfp_reg_8lselu_stack_JTT_v222_2179cb69e6a58dd5926caf1.csv',
	'ecfp_reg_8lselu_stack_JTT_v224_de63364caceffe26e169240.csv',
	'ecfp_reg_8lselu_stack_JTT_v237_7ab256ee5e22e90c2814054.csv',
	'ecfp_reg_8lselu_stack_JTT_v240_29818e37e3b3618709374c2.csv',
	'ecfp_reg_8lselu_stack_JTT_v248_6eb13331cc79d629433489d.csv',
	'ecfp_reg_8lselu_stack_JTT_v251_ce25adf18e91efbe1f62798.csv',
	'ecfp_reg_8lselu_stack_JTT_v255_7ad21c176b8679447c2d168.csv',
	'ecfp_reg_8lselu_multi_v285_e801706d407464efebec046.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v348_d013068d37f6be474582c2e.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v387_4840b222d93c4705947122e.csv',
	'ecfp_avg_ensemble_v403_f0cf36f0868f3f3d07398c8.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v430_4736a0bb2938ac7915854e8.csv',
	'ecfp_reg_8lselu_v130_11e1aa8015b88969c17244e.csv',
	'ecfp_reg_8lselu_v132_11fa2099d53ad7cc0abde6c.csv',
	'ecfp_reg_8lselu_stack_v144_bf89a417bc2427a1cf5bfec.csv',
	'ecfp_reg_8lselu_stack_v152_f90e7fbae19c78b52b8be3f.csv',
	'ecfp_reg_8lselu_stack_v194_5ca9b30400a3ed30c8e4201.csv',
	'ecfp_reg_8lselu_stack_v198_273a77555f5a4686b1b84f2.csv',
	'ecfp_reg_8lselu_stack_JTT_v250_501ed78c136a37ac81ff355.csv',
	'ecfp_reg_8lselu_stack_JTT_v252_9fc9321c577afe57f9ffba4.csv',
	'ecfp_reg_8lselu_multi_v345_8703e7f6bc44606f8dc3e10.csv',
	'ecfp_reg_8lselu_multi_v347_31b3ccb62c5fc9bb2edddf8.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v407_b3d2d0cb2da1afac43e8f25.csv',
	'ecfp_reg_8lselu_multi_v412_80452eb271606e4f2c00f72.csv',
	'ecfp_reg_8lselu_multi_v415_71ca3b670cf15f74bd01199.csv',
	'ecfp_reg_8lselu_v109_3efb4b3b84cc87e84f5ee1f.csv',
	'ecfp_reg_8lselu_stack_v143_96cafa14e3b65e5d97cf178.csv',
	'ecfp_reg_8lselu_stack_v160_1117e931307da3cdb624320.csv',
	'ecfp_reg_8lselu_stack_v171_ffd6f7d508f23b7388a738f.csv',
	'ecfp_reg_8lselu_stack_v210_9d40c364f74cb39246e3501.csv',
	'ecfp_reg_8lselu_multi_v291_33c2a737b1c72b6167e19e9.csv',
	'ecfp_reg_8lselu_multi_v314_d26d21e4acf7b99f3bbd1f0.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v368_97286bc7c72e73d090a898c.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v383_680253c8ce7a6216adebf1e.csv',
	'ecfp_avg_ensemble_v405_260a61c97f33a3b674ef111.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v419_03e65520ca03e3a7ef120e1.csv',
	'ecfp_reg_8lselu_multi_ensemble_low_v432_a126037cf8fda14ea1b3021.csv',
	'ecfp_reg_8lselu_v107_c57a7bb6e8bab24abb31d7a.csv']


LB_scores = [0.557, 0.557, 0.557, 0.557, 0.557, 0.558, 0.558, 0.558, 0.558, 0.559, 0.561, 0.561, 0.563, 0.563, 0.564, 0.564,
	0.564, 0.564, 0.565, 0.565, 0.565, 0.565, 0.565, 0.565, 0.565, 0.565, 0.565, 0.566, 0.566, 0.566, 0.566, 0.567,
	0.567, 0.567, 0.567, 0.567, 0.567, 0.567, 0.567, 0.568, 0.568, 0.568, 0.568, 0.568, 0.568, 0.568, 0.568, 0.568,
	0.568, 0.568, 0.568, 0.568, 0.568, 0.568, 0.569, 0.569, 0.569, 0.569, 0.569, 0.569, 0.569, 0.569, 0.569, 0.569,
	0.569, 0.569, 0.569, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.571]


n_models = 80
# Weights
equal_weight = False
power = 3

if equal_weight:
	# Equal weight scheme
	weights = [1/n_models] * n_models
	weights = np.array(weights)
else:
	scores = np.array(LB_scores)
	scores = scores ** power
	scores = 1 - scores
	# Calculate weights
	weights = scores / np.sum(scores)

for i in range(n_models):
	if i == 0:
		model = pd.read_csv(models_list[i])
		average_model = weights[i] * model.iloc[:, 1:]
	else:
		model = pd.read_csv(models_list[i])
		average_model = average_model + (weights[i] * model.iloc[:, 1:])

# Averaged
ensembled = model
ensembled.iloc[:, 1:] = average_model

# Hash for averaged model
truncated_hash = hashlib.sha256(os.urandom(23)).hexdigest()[:23]
print(truncated_hash)

# Save
submission_path = '/Output/ecfp_avg_ensemble_80_LB_p3_v491_' + truncated_hash + '.csv'
ensembled.to_csv(submission_path, index=False)
