from initialization import *
def initialize_extraction():
    """
    The function is to initialize the radiomic feature extractor
    return: radiomic feature extractor
    """
    settings = {'binWidth': 20, 'sigma': [1, 2, 3]}

    # Instantiate the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()

    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)  # Still the default parameters
    print('Enabled features:\n\t', extractor.enabledFeatures)  # Still the default parameters
    print('Enabled filters:\n\t', extractor.enabledImagetypes)

    return extractor

def extract_features(extractor, cases, image_dir, mask_dir, output_dir):
    """
    :param extractor: radiomic extractor defined before
    :param cases: a list of cases with valid image and corresponding mask
    :param image_dir: the path of folder containing image files
    :param mask_dir: the path of folder containing mask files
    :param output_dir: the path of folder for storage of extracted features in excel files
    :return: a list of cases in which the extraction failed
    """    """
    :param extractor: radiomic extractor defined before
    :param cases: a list of cases with valid image and corresponding mask
    :param image_dir: the path of folder containing image files
    :param mask_dir: the path of folder containing mask files
    :param output_dir: the path of folder for storage of extracted features in excel files
    :return: a list of cases in which the extraction failed
    """

    failed_case = []

    for case in cases:

        case_name = case.split(".")[0]
        cond_feature = os.path.exists(f"{output_dir}/{case_name}.xlsx")

        if not cond_feature:  # skip cases already extracted
            print(f"Processing {case_name}", end="\r")

            image_path = f'{image_dir}/{case}'
            roi_path = f'{mask_dir}/{case}'

        try:
            result = extractor.execute(image_path, roi_path)
            data = pd.DataFrame(columns=result.keys())
            data = data.iloc[:, 22:]
            for featureName in data.columns:
                data.loc[0, featureName] = result[featureName]
            data["Case"] = str(case_name)
            data.to_excel(f"{output_dir}/{case_name}.xlsx", index=False)
        except:
            failed_case.append(case_name)

    if failed_case != []:
        print(f"Totoally, {len(failed_case)} cases failed!")
        print(f"Failed cases: {failed_case}")

    return failed_case


def concatenate_features(feature_path):
    """
    :param feature_path: the path of folder for storage of individual excel files for extracted features
    :return: concatenated dataframe of all individual radiomic feature file
    """
    # The function is to concatenate individual radiomics file in xlsx into a dataframe
    # The input is the path where the individual radiomics file is stored
    # The output is a dataframe
    ds = []
    xlsx_list = [x for x in os.listdir(feature_path) if
                 x.endswith("xlsx") and not x.startswith("._")]  # avoid error in Mac OS

    for i in xlsx_list:
        i_name = i.split(".")[0]
        print(f"Processing: {i_name}")
        df_i = pd.read_excel(f'{feature_path}/{i}')
        ds.append(df_i)

    df = pd.concat(ds)

    return df
