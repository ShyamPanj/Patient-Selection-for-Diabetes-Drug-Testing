import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    ndc_df.rename(columns={'NDC_Code':'ndc_code'}, inplace=True)
    df=df.merge(ndc_df,on='ndc_code')
    df.rename(columns={'Non-proprietary Name':'generic_drug_name'},inplace=True)
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    df.sort_values(by=['patient_nbr','encounter_id'],inplace=True)
    first_encounter_df=reduce_dim_df.groupby(by='patient_nbr',as_index=False).first()
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    idx=np.random.randint(0,len(df)-1,len(df))
    unique_patients=df.patient_nbr.unique()
    patient_train=unique_patients[0:int(0.60*len(df))]
    patient_val=unique_patients[int(0.60*len(df)):int(0.60*len(df)+0.20*len(df))]
    patient_test=unique_patients[int(0.60*len(df)+0.20*len(df)):]
    train=df[df['patient_nbr'].isin(patient_train)]
    validation=df[df['patient_nbr'].isin(patient_val)]
    test=df[df['patient_nbr'].isin(patient_test)]
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
    cat_column=tf.feature_column.categorical_column_with_vocabulary_file(c, vocab_file_path,default_value='00')
    tf_categorical_feature_column = tf.feature_column.indicator_column(cat_column)
    output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    tf_numeric_feature=tf.feature_column.numeric_column(col, default_value=default_value, dtype=tf.dtypes.float32, normalizer_fn=lambda x: (x-MEAN)/ STD)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction
