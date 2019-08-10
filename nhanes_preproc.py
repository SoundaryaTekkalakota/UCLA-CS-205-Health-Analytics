import pdb
import glob
import copy
import os
import pickle
import joblib
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.feature_selection
from sklearn.preprocessing import KBinsDiscretizer
import pdb
class FeatureColumn:
    def __init__(self, category, field, preprocessor, args=None, cost=None):
        self.category = category
        self.field = field
        self.preprocessor = preprocessor
        self.args = args
        self.data = None
        self.cost = cost

class NHANES:
    def __init__(self, db_path=None, columns=None):
        self.db_path = db_path
        self.columns = columns # Depricated
        self.dataset = None # Depricated
        self.column_data = None
        self.column_info = None
        self.df_features = None
        self.df_targets = None
        self.costs = None

    def process(self):
        df = None
        cache = {}
        # collect relevant data
        df = []
        for fe_col in self.columns:
            sheet = fe_col.category
            field = fe_col.field
            data_files = glob.glob(self.db_path+sheet+'/*.XPT')
            df_col = []
            for dfile in data_files:
                print(80*' ', end='\r')
                print('\rProcessing: ' + dfile.split('/')[-1], end='')
                # read the file
                if dfile in cache:
                    df_tmp = cache[dfile]
                else:
                    df_tmp = pd.read_sas(dfile)
                    cache[dfile] = df_tmp
                # skip of there is no SEQN
                if 'SEQN' not in df_tmp.columns:
                    continue
                #df_tmp.set_index('SEQN')
                # skip if there is nothing interseting there
                sel_cols = set(df_tmp.columns).intersection([field])
                if not sel_cols:
                    continue
                else:
                    df_tmp = df_tmp[['SEQN'] + list(sel_cols)]
                    df_tmp.set_index('SEQN', inplace=True)
                    df_col.append(df_tmp)

            try:
                df_col = pd.concat(df_col)
            except:
                #raise Error('Failed to process' + field)
                raise Exception('Failed to process' + field)

            df.append(df_col)


        df = reduce(lambda left, right : pd.merge(left,right, on = 'SEQN', how='outer'), df)
        df = df.groupby(df.index).first()

        df_proc = []
        for fe_col in self.columns:
            field = fe_col.field
            fe_col.data = df[field].copy()
            # do preprocessing
            if fe_col.preprocessor is not None:
                prepr_col = fe_col.preprocessor(df[field], fe_col.args)
            else:
                prepr_col = df[field]
            # handle the 1 to many
            if (len(prepr_col.shape) > 1):
                fe_col.cost = [fe_col.cost] * prepr_col.shape[1]
            else:
                fe_col.cost = [fe_col.cost]
            df_proc.append(prepr_col)
        self.dataset = pd.concat(df_proc, axis=1)
        return self.dataset


# Preprocessing functions
def impute_onehot(df_col, args=None):
    df_col[pd.isna(df_col)] = df_col.mean()
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')


def preproc_onehot(df_col, args=None):
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_real(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    # statistical normalization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_impute(df_col, args=None):
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    return df_col

def preproc_mode(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mode()
    return df_col

def preproc_bin(df_col,args=None):
    #Do binning
    if args is None:
        args = {'n_bins':5,'replace_na':preproc_impute}
    if 'replace_na' not in args:
        args['replace_na'] = preproc_impute
    df_col = args['replace_na'](df_col)
    bins = np.linspace(df_col.min()-1,df_col.max()+1,args['n_bins'])
    return pd.cut(df_col, bins, labels=False)

def preproc_mode_onehot(df_col,args=None):
    if args is None:
        args={'cutoff':np.inf}
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_cut(df_col, bins):
    # limit values to the bins range
    df_col = df_col[df_col >= bins[0]]
    df_col = df_col[df_col <= bins[-1]]
    return pd.cut(df_col.iloc[:,0], bins, labels=False)

def preproc_dropna(df_col, args=None):
    df_col.dropna(axis=0, how='any', inplace=True)
    return df_col



# Dataset loader
class Dataset():
    """
    Dataset manager class
    """
    def  __init__(self, data_path=None):
        """
        Class intitializer.
        """

        if data_path == None:
            self.data_path = './run_data/'
        else:
            self.data_path = data_path

        self.features = None
        self.targets = None
        self.costs = None

    def load_cancer(self, opts=None):
        columns = [
            # TARGET: Cancer or malignancy ?
            FeatureColumn('Questionnaire', 'MCQ220',
                                    None, None),
            # Gender
            FeatureColumn('Demographics', 'RIAGENDR',
                                 None, None),
            # Age at time of screening
            FeatureColumn('Demographics', 'RIDAGEYR',
                                 preproc_real, None),
            FeatureColumn('Demographics', 'RIDRETH3',
                                 preproc_real, None),
            # Race/ethnicity
            FeatureColumn('Demographics', 'RIDRETH1',
                                 preproc_real, None),
            # Annual household income
            FeatureColumn('Demographics', 'INDHHINC',
                                 preproc_real, {'cutoff':11}),
            # BMI
            FeatureColumn('Examination', 'BMXBMI',
                                 preproc_real, None),
            # Waist
            FeatureColumn('Examination', 'BMXWAIST',
                                 preproc_real, None),
            # Height
            FeatureColumn('Examination', 'BMXHT',
                                 preproc_real, None),
            # Weight
            FeatureColumn('Examination', 'BMXWT',
                                 preproc_real, None),
            # Total Cholesterol
            FeatureColumn('Laboratory', 'LBXTC',
                                 preproc_real, None),
            # Alcohol consumption
            FeatureColumn('Questionnaire', 'ALQ101',
                                 preproc_real, {'cutoff':2}),
            FeatureColumn('Questionnaire', 'ALQ120Q',
                                 preproc_real, {'cutoff':365}),
            # Vigorous work activity
            FeatureColumn('Questionnaire', 'PAQ605',
                                 preproc_real, {'cutoff':2}),
            FeatureColumn('Questionnaire', 'PAQ620',
                                 preproc_real, {'cutoff':2}),
            FeatureColumn('Questionnaire', 'PAQ180',
                                 preproc_real, {'cutoff':4}),

            FeatureColumn('Questionnaire', 'PAD615',
                                 preproc_real, {'cutoff':780}),
            # Doctor told overweight (risk factor)
            FeatureColumn('Questionnaire', 'MCQ160J',
                                 preproc_real, {'cutoff':2}),
            # Sleep
            FeatureColumn('Questionnaire', 'SLD010H',
                                 preproc_real, {'cutoff':12}),
            # Smoking
            FeatureColumn('Questionnaire', 'SMQ020',
                                 preproc_onehot, {'cutoff':3}),
            FeatureColumn('Questionnaire', 'SMD030',
                                 preproc_real, {'cutoff':80}),

            # How often add salt to food at table
            FeatureColumn('Dietary', 'DBD100',
                                 preproc_onehot, {'cutoff':3}),
            # Total plain water drank yesterday (gm)
            FeatureColumn('Dietary', 'DR1_320Z',
                                 preproc_impute, None),

            # Annual household income
            FeatureColumn('Demographics', 'INDHHIN2',
                                  preproc_impute, None),

            # Ratio of family income to poverty
            FeatureColumn('Demographics', 'INDFMPIR',
                                  preproc_real, None),
            # Sodium (mg)
            FeatureColumn('Dietary', 'DR2TSODI',
                                  preproc_real, None),
            # Number of rooms in home
            FeatureColumn('Questionnaire', 'HOD050',
                                  preproc_onehot, {'cutoff':13}),

            FeatureColumn('Questionnaire', 'SMQ040',
                                  preproc_real, {'cutoff':3}),
            # Arsenic - Total - Urine
            FeatureColumn('Laboratory', 'URXUAS',
                                 preproc_bin, {'n_bins':100,'replace_na':preproc_impute} ),

            # Lead, Cadmium, Total Mercury, Selenium & Manganese - Blood
            FeatureColumn('Laboratory', 'LBDBPBSI',
                                 preproc_real, None),
            # Lead, Cadmium, Total Mercury, Selenium & Manganese - Blood
            FeatureColumn('Laboratory', 'LBDBCDSI',
                                 preproc_real, None),
            # Lead, Cadmium, Total Mercury, Selenium & Manganese - Blood
            FeatureColumn('Laboratory', 'LBDTHGSI',
                                 preproc_bin, {'n_bins':50,'replace_na':preproc_impute}),
            # Lead, Cadmium, Total Mercury, Selenium & Manganese - Blood
            FeatureColumn('Laboratory', 'LBDBSESI',
                                 preproc_real, None),
            # Lead, Cadmium, Total Mercury, Selenium & Manganese - Blood
            FeatureColumn('Laboratory', 'LBDBMNSI',
                                 preproc_real, None),
            # Mercury: Inorganic, Ethyl and Methyl – Blood
            FeatureColumn('Laboratory', 'WTSH2YR',
                                 preproc_real, None),
            # Mercury: Inorganic, Ethyl and Methyl – Blood
            FeatureColumn('Laboratory', 'LBDIHGSI',
                                 preproc_real, None),
            # Human Papillomavirus (HPV) - Oral Rinse
            FeatureColumn('Laboratory', 'ORXHPV',
                                 preproc_real, {'cutoff':2}),
            # Oral Health
            FeatureColumn('Examination', 'OHDEXSTS',
                                         preproc_real, {'cutoff':2}),
            # Urine
            FeatureColumn('Laboratory', 'URXUMA',
                                         preproc_real, None),
            FeatureColumn('Laboratory', 'URDACT',
                                         preproc_real, None),

            # Apolipoprotein
            FeatureColumn('Laboratory', 'LBDAPBSI',
                                         preproc_impute, None),
            # Sunburn
            FeatureColumn('Questionnaire', 'DEQ038G',
                                         preproc_real, {'cutoff':2}),

            # Health Insurance
            FeatureColumn('Questionnaire', 'HIQ011',
                                 preproc_real, {'cutoff':2}),

            # Mother Smoked
            FeatureColumn('Questionnaire', 'ECQ020',
                                         preproc_real, {'cutoff':2}),
            # Diet
            FeatureColumn('Questionnaire', 'DBQ700',
                                 preproc_onehot, {'cutoff':5}),
            # Heapatitis B
            FeatureColumn('Laboratory', 'LBXHBC',
                                         preproc_real, {'cutoff':2}),
            # Education level - Child
            FeatureColumn('Demographics', 'DMDEDUC3',
                                 preproc_real, {'cutoff':15}),

            # Country of birth
            FeatureColumn('Demographics', 'DMDBORN4',
                                 preproc_real, {'cutoff':2}),
            # How often do you snore?
            FeatureColumn('Questionnaire', 'SLQ030',
                                 preproc_real,  {'cutoff':6}),
            # Received Hepatitis A vaccine
            FeatureColumn('Questionnaire', 'IMQ011',
                                 preproc_real, {'cutoff':3}),

            # Received Hepatitis B 3 dose series
            FeatureColumn('Questionnaire', 'IMQ020',
                                 preproc_real, {'cutoff':3}),
            #Blood pressure
             FeatureColumn('Questionnaire', 'BPQ020',
                                         preproc_real, {'cutoff':2}),
            #Has doctor told you that you High Cholesterol level
             FeatureColumn('Questionnaire', 'BPQ080',
                                         preproc_real, {'cutoff':2}),
            #Has doctor told you that you have Diabetes
             FeatureColumn('Questionnaire', 'DIQ010',
                                         preproc_real, {'cutoff':3}),
            #Stomach or intestinal illness
            FeatureColumn('Questionnaire', 'HSQ510',
                                         preproc_real, {'cutoff':2}),
            #Fluoride, water (mg/L) average 2 values
            FeatureColumn('Laboratory', 'LBDWFL',
                                         preproc_real, None),
           #plasma glucose values
             FeatureColumn('Laboratory', 'LBXGLU',
                                         preproc_real, None),
           # air quality bad
            FeatureColumn('Questionnaire', 'PAQ685',
                                         preproc_real,  {'cutoff':3}),
            #ever been told that you have a sleep disorder
            FeatureColumn('Questionnaire', 'SLQ060',
                                         preproc_real, {'cutoff':2}),
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        fe_cols = df.drop(['MCQ220'], axis=1)
        features = fe_cols.values
        target = df['MCQ220'].values
            # remove nan labeled samples
        inds_valid = ~ np.isnan(target)
        features = features[inds_valid]
        target = target[inds_valid]

            # Put each person in the corresponding bin
        targets = np.full((target.shape[0]), 3)
        targets[target == 1] = 0 # yes cancer
        targets[target == 2] = 1 # no cancer

           # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
                [item for sublist in self.costs for item in sublist])
