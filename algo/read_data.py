

import numpy as np
import pandas as pd


from .groups import Group

#import sys, os
#sys.path.append(os.path.dirname(file))

def df_exceptions_dict(df):
    """
    переводит таблицу в словарь из запретов -- у каждого объекта запрещаются свои слоты
    """
    
    answer = {}
    indexes = np.arange(1, df.shape[1])

    for i in range(df.shape[0]):
        answer[df.iloc[i,0]] = indexes[df.values[i,1:] == 0]
    
    return {key: list(val) for key, val in answer.items()}


def df_to_better_view(df):
    """
    меняет значения ячеек таблицы на более симпатичные
    """
    df = df.astype(str)
    df[df == '0.0'] = 'нельзя использовать'
    df[df == '0'] = 'нельзя использовать'
    df[df == '-1.0'] = ''
    df[df == '-1'] = ''

    return df








def read_slots_table(file_name = '../data/Diss.xlsx'):
    
    df = pd.read_excel(file_name, sheet_name = 'Timeslots').iloc[1:,:]
    
    df.columns = df.iloc[0, :]
    
    return df.iloc[1:, :]



def read_teachers_schedule(file_name = '../data/Diss.xlsx'):

    df = pd.read_excel(file_name, sheet_name = 'Schedule teachers').iloc[3:,1:].fillna(-1)
    
    df.columns = [df.iloc[0,0]] + [str(int(v)) for v in df.iloc[0,1:]]
    
    df = df.iloc[1:,:]
    
    return df


def read_places_schedule(file_name = '../data/Diss.xlsx'):

    df = pd.read_excel(file_name, sheet_name = 'Schedule places').iloc[2:,1:].fillna(-1)
        
    df.columns = [df.iloc[0,0]] + [str(int(v)) for v in df.iloc[0,1:]]
        
    df = df.iloc[1:,:]
    
    return df


def read_groups_schedule(file_name = '../data/Diss.xlsx'):
    
    df = pd.read_excel(file_name, sheet_name = 'Schedule groups').iloc[3:,1:].fillna(-1)
        
    df.columns = [df.iloc[0,0]] + [str(int(v)) for v in df.iloc[0,1:]]
        
    df = df.iloc[1:,:]
    
    return df





def read_teachers(file_name = '../data/Diss.xlsx'):
    
    df = pd.read_excel(file_name, sheet_name = 'Teachers')
    
    data = {key: [val.strip() for val in val.split(';') if val] for key, val in zip(df.iloc[:,0], df.iloc[:,1])}
    
    return data


def read_places(file_name = '../data/Diss.xlsx'):
    
    df = pd.read_excel(file_name, sheet_name = 'Places')
    
    places = df.iloc[:,0]
    capacitys = df.iloc[:,1]
    contains = [set([]) if v is np.nan else set([vl.strip() for vl in v.split(',') if vl]) for v in df.iloc[:,2]] 

    
    return dict(zip(places, capacitys)), dict(zip(places, contains))


def read_subjects(file_name = '../data/Diss.xlsx'):
    
    df = pd.read_excel(file_name, sheet_name = 'Subjects')
    
    subjects = df.iloc[:,0]
    reqs = [set([]) if v is np.nan else set([vl.strip() for vl in v.split(',') if vl]) for v in df.iloc[:,2]]
    difc =  df.iloc[:,3]  
    hours = df.iloc[:,4]

    
    return dict(zip(subjects, reqs)), dict(zip(subjects, difc)), dict(zip(subjects, hours))


def read_groups(file_name = '../data/Diss.xlsx'):
    
    df = pd.read_excel(file_name, sheet_name = 'Groups')
    
    code = df.iloc[:,0]
    names = df.iloc[:,1]
    childs = [None if v is np.nan else [int(vl.strip()) for vl in v.split() if vl] for v in df.iloc[:,2]] 
    capacitys =  df.iloc[:,3]  


    return dict(zip(code, names)), dict(zip(code, capacitys)), [Group(id_, childs_) for id_, childs_ in zip(code, childs)]




def read_all_data(file_name = '../data/Diss.xlsx'):

    answer = {}

    for name, reader in zip(('teacher', 'places', 'groups'), (read_teachers_schedule, read_places_schedule, read_groups_schedule)):
        answer[f'{name}_schedule'] = reader(file_name)
        answer[f'{name}_exceptions'] = df_exceptions_dict(answer[f'{name}_schedule'])
        answer[f'{name}_schedule'] = df_to_better_view(answer[f'{name}_schedule'])

    answer['teachers'] = read_teachers(file_name)

    answer['places2counts'], answer['places2contain'] = read_places(file_name)

    answer['subjects2reqs'], answer['subjects2diffc'], answer['subjects2hours'] = read_subjects(file_name)

    answer['groups2names'], answer['groups2counts'], answer['all_groups'] = read_groups(file_name)


    answer['table_slots'] = read_slots_table(file_name)


    return answer





if __name__ == '__main__':

    result = read_all_data()




















