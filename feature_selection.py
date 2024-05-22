from initialization import *
from matplotlib.ticker import PercentFormatter
#The feature selection includes unsupervised and supervised LASSO-based feature selections.
def select_feature_corr(df_train, df_test,radiomics_feature,correlation_threshold = 0.85):
	"""
	The function is to perfrom unsupervised feature selection by excluding one of the highly correlated features.
	:param df_train: training data
	:param df_test: test data
	:param radiomics_feature: column names for radiomic features
	:param correlation_threshold: threshold to define highly-correlated features
	return: processed training data, processed test data, list of retained features
	"""

	# create correlation  matrix
	corr_matrix = df_train[radiomics_feature].corr().abs()

	# select upper traingle of correlation matrix
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

	# Find index of columns with correlation greater than 0.95
	to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
	non_related_features = [column for column in upper.columns if not column in to_drop]

	# drop the columns
	df_train = df_train.drop(df_train[to_drop], axis=1)
	df_test = df_test.drop(df_test[to_drop], axis=1)
	print(f"Drop {len(to_drop)} features")
	print(f"Keep {len(non_related_features)} features")

	return df_train,df_test,non_related_features

#----LASSO-based method---#

def LASSO(df_input, y_name,random_state, test_size=0.2):
	"""
	The function is to set up a pipeline for supervised LASSO-based feature selection.
	:param df_input: data for LASSO
	:param y_name: column name for Y label
	:param test_size: percentage for test data in LASSO regression
	:param random_state: random state for data splitting
	return: feature names and corresponding coefficient
	"""

	y = df_input[y_name]

	list_to_drop = [y_name]
	X = df_input.drop(list_to_drop, axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

	model = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)

	model.fit(X_train, y_train)
	coef = model.coef_

	imp_features = pd.Series(X_train.columns)[list(coef != 0)]

	X_train_LASSO = X_train[imp_features]
	X_test_LASSO = X_test[imp_features]
	return imp_features, coef[list(coef != 0)]

def bootstrap_LASSO(df_train,non_related_features,output_dir, y_name = "Origin_num",cycle_number = 100):
	"""
	The function is to perfrom supervised LASSO-based feature selection by bootsraping training data.
	:param df_train: training data for LASSO-based feature selection
	:param non_related_features: column name unrelated features from unsupervised feature selection
	:param output_dir: path to save output
	:param y_name: column name for Y label
	"""
	# repeat for 100 times for demonstration purpose. Normally, I adopted 1000 for actual calculation
	df_features_lasso = df_train[non_related_features + [y_name]]

	for i in range(cycle_number):
		output_name = f'Lasso_seed_{i}.xlsx'
		if not os.path.exists(os.path.join(output_dir, output_name)):
			output = {}
			output['feature'] = []
			output['coef'] = []

			imp_features, _coef = LASSO(df_features_lasso, y_name,i,test_size=0.2)
			output['feature'] = imp_features
			output['coef'] = _coef
			df_i = pd.DataFrame.from_dict(output)
			df_i.to_excel(os.path.join(output_dir, output_name), index=False)

def summarize_lasso_result(lasso_output_dir):
	"""
	The function is to summarize results from LASSO-based feature selection.
	The mean, standard deviation and percentage of missing cycle are calculated.
	:param lasso_output_dir: folder path for saving individual LASSO-based feature selection
	return concatenated and processed dataframe for LASSO-based feature selection
	"""
	lasso_list = [x for x in os.listdir(lasso_output_dir) if x.startswith("Lasso_seed_")]
	df_list = []
	for i in lasso_list:
		i_file = pd.read_excel(f'{lasso_output_dir}/{i}')
		length = i_file.shape[0]
		i_file['cycle'] = [i[:-5] for x in range(int(length))]
		df_list.append(i_file)
	df_final = pd.concat(df_list)
	df_wide = pd.pivot(df_final, index=['feature'], columns='cycle', values='coef')

	df_wide['mean'] = df_wide.mean(axis=1)
	df_wide['sd'] = df_wide.std(axis=1)
	df_wide.sort_values(by=['mean'])
	cycle_number = len(np.unique(df_final['cycle']))
	df_wide['Missing_cycle(%)'] = [x / (cycle_number) for x in df_wide.isnull().T.sum().T]
	cols = ['mean', 'Missing_cycle(%)', 'sd']
	df_wide = df_wide[cols + [c for c in df_wide.columns if c not in cols]]
	df_wide.reset_index(inplace=True)
	df_wide.to_excel(os.path.join(lasso_output_dir, f'compile_{len(df_list)}.xlsx'), index=False)

	return df_wide

def get_selected_feature_by_threshold(lasso_output_dir,threshold = 0.2):
	"""
	The function is to get selected features by LASSO by thresholding
	:param lasso_output_dir: folder path for saving compiled LASSO-based feature selection
	:param threshold: threshold for selecting features
	return: 1. dataframe for compiled LASSO-based feature selection
			2. names of selected features
	"""
	file_name = [x for x in os.listdir(lasso_output_dir) if x.startswith("compile")][0]
	file_path = f"{lasso_output_dir}/{file_name}"
	df = pd.read_excel(file_path)
	selected_features = df[df["Missing_cycle(%)"] < threshold]["feature"]
	return df, selected_features

def plot_feature_importance(df,output_dir,threshold = 0.2):
	"""
	The function is to plot relative feature importance
	:param df: dataframe for compiled LASSO-based feature selection
	:param output_dir: folder path to save the plot
	:param threshold: threshold for selecting features to be plotted
	"""
	# data preparation
	df_plot = df.loc[df["Missing_cycle(%)"] < threshold, ["mean", 'feature']]
	df_plot.reset_index(inplace=True)

	#If more than 10 features are eligible, we plot only the top 10 features
	df_plot = df_plot.sort_values(by=['mean'], ascending=False).iloc[:10, ]

	feature_names = df_plot['feature'].str.replace('_', '\n')
	importance = df_plot['mean']
	importance = np.abs(importance) / np.sum(np.abs(importance))
	indices = np.argsort(importance)

	#Plotting
	plt.figure(figsize=(10, np.ceil(len(df_plot) * 1.5)))  ##
	plt.title(f'Relative feature importance', fontsize=40, fontname="Arial", fontweight='semibold', pad=18)
	plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
	plt.barh(range(len(indices)), importance, color='#219ebc', align='center')
	# plt.barh(range(len(indices)), importance, color='#EAEAF2', align='center')
	x_max = np.ceil(max(importance) * 11) / 10
	plt.xlim(0, x_max)  ##
	for index, value in enumerate(importance):
		plt.text(value, index, " {:.1%}".format(value), fontsize=26)

	feature = feature_names
	feature_name_to_show = feature

	# plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
	plt.xticks(fontname="Arial", fontsize=20)
	plt.yticks(range(len(indices)), [i for i in feature_name_to_show], fontname="Arial", fontsize=20)
	plt.gca().invert_yaxis()
	plt.savefig(f"{output_dir}/relative_importance.jpg", dpi=300, bbox_inches='tight')
	plt.show()