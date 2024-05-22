from initialization import *
import pandas as pd

def get_label(df):
	"""
	The function is to get Y label from the case names and code the Y label as numbers.
	Additionally, a column named "Case ID" is also added.
	:param df: concatenated dataframe of radiomic features
	:return: 1. dataframe with Y label, numeric Y label and "Case ID" added
			 2. dataframe documenting the mapping between Y label in string and Y label in numbers
	"""

	df["Origin"] = df["Case"].str.extract('(?:[^a-zA-Z]|^)([a-zA-Z]+)(?:[^a-zA-Z]|$)')
	df["Origin"] = df["Origin"].str.rstrip('lr')   #based on the readme, the "l" and "r" are treated as the same.
	df["Origin"].replace('live', 'liver', inplace=True)
	# add case IDs
	num_rows = len(df)
	case_ids = ['IM' + str(i + 1) for i in range(num_rows)]
	df['Case ID'] = case_ids

	# name Y as numbers
	y_rename_dict = {value: key for key, value in dict(list(enumerate(df["Origin"].unique()))).items()}

	df['Origin_num'] = df['Origin'].map(y_rename_dict)

	df_y_rename = pd.DataFrame(y_rename_dict, index=[0])
	return df, y_rename_dict,df_y_rename

def add_ID_predict(df):
	"""
	The function, called only in future application, is add an column of Case ID.
	:param df: concatenated dataframe of radiomic features
	:return: dataframe with a column named Case ID added.
	"""

	num_rows = len(df)
	case_ids = ['IM' + str(i + 1) for i in range(num_rows)]
	df['Case ID'] = case_ids

	return df

def visualize_Y(df,y_name,output_dir):
	"""
	The function is create a bar plot the distribution of Y label.
	:param df: concatenated dataframe of radiomic features
	:param y_name: the name of column of Y label
	:param output_dir: the folder to save the bar plot
	"""
	data_counts = df[y_name].value_counts().sort_index()

	# Create a bar plot
	plt.bar(data_counts.index, data_counts.values, color='green')
	plt.title('Distribution of Organ')
	plt.xlabel('Origin')
	plt.ylabel('Count')
	plt.xticks(range(len(df["Origin_num"].unique())))
	plt.savefig(f"{output_dir}/Y_distribution.jpg")
	plt.show()

# Make sure every category appear in both training and test datasets
def stratified_split(df, stratify_col, test_size=0.2):
	"""
	The function is to randomly split the dataset into training and test data stratified by Y label.
	:param df: concatenated dataframe of radiomic features
	:param stratify_col: column name based on which the stratification is perfromed
	:param test_size: the percentage of test dataset
	:return: training data and test data
	"""

	train_dfs = []
	test_dfs = []

	for category in df[stratify_col].unique():
		category_df = df[df[stratify_col] == category]
		category_df.reset_index(drop=True, inplace=True)
		test_sample = category_df.sample(frac=test_size, random_state=42)
		train_sample = category_df.drop(test_sample.index)

		train_dfs.append(train_sample)
		test_dfs.append(test_sample)

	train_df = pd.concat(train_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
	test_df = pd.concat(test_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

	return train_df, test_df

def get_radiomic_feature_name(df):
	"""
	The function is get the columns for radiomic features automatically by regular expression.
	:param df: concatenated dataframe of radiomic features
	return: column names for radiomic features
	"""
	radiomics_feature = df.filter(regex=r'^[^_]*_[^_]*_[^_]*$').columns.to_list()
	return radiomics_feature

def normalize_data(df_train,df_test,radiomics_feature):
	"""
	The function is to train a scaler based on training data and to normalize both training data and test data by scaler.
	:param df_train: training data
	:param df_test: test data
	:param radiomics_feature: column names for radiomic features
	return: normalized training data, normalized test data, scaler
	"""
	scaler = StandardScaler()
	scaler.fit(df_train[radiomics_feature])

	df_train[radiomics_feature] = scaler.transform(df_train[radiomics_feature])
	df_test[radiomics_feature] = scaler.transform(df_test[radiomics_feature])

	return df_train,df_test,scaler

def normalize_data_predict(df,radiomics_feature,scaler):
	"""
	The function is to normalize data by scaler trained before.
	:param df: dataframe of extracted features
	:param radiomics_feature: column names for radiomic features
	:param scaler: trained scaler for normalization
	return: normalized data
	"""
	df[radiomics_feature] = scaler.transform(df[radiomics_feature])

	return df