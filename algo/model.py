

import numpy as np
import pandas as pd



from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Population_initializer # for creating better start population

from DiscreteHillClimbing import Hill_Climbing_descent


# перевод сложности в метрику
def difficult2score(value):
    if value > 20: return 10*(value-20) + value
    return value


class Optimizator:

    def __init__(self, result, all_combinations, slots):

        self.result = result
        self.combinations = all_combinations
        self.slots = slots

    
    def set_evaluators(self):

        needed_df = self.combinations.loc[:, ['subject', 'group']].values # для оценки надо знать только предмет и группу, остальные вещи вообще не важны

        subjects2hours = self.result['subjects2hours']
        subjects2diffs = self.result['subjects2diffc']

        all_subjects = list(subjects2diffs.keys()) + [i for i in range(1, 7)]

        all_groups = self.result['all_groups']
        group_keys = {group.id: i for i, group in enumerate(all_groups)} # чтобы по айди группы брать номер группы в списке
        last_groups = [i for i, g in enumerate(all_groups) if g.last] # группы без детей - по ним метрики и считаем

        timeslot2day = np.zeros(len(self.slots), dtype = int) # соответствие между номером слота и днём
        for i in range(1, 7):
            timeslot2day[(i-1)*7:i*7] = i

        slot_indexes = np.arange(len(self.slots))



        def eval(vector):

            nonlocal all_groups

            vec = vector.astype(int)
            mask = vec >= 0

            if mask.sum() == vec.size: return 100_000


            df = needed_df[vec[mask],:]

            for g in all_groups: # обнуляю счетчики
                g.set_subjects(all_subjects)


            for i, subj, group in zip(slot_indexes[mask], df[:,0], df[:,1]):

                dct = {
                    subj: 1, # +1 час по нужному предмету
                    timeslot2day[i]: subjects2diffs[subj] # сложность в этом дне
                }

                all_groups[group_keys[group]].add_score(dct, all_groups)

            
            total = 0

            for number_g in last_groups:

                g = all_groups[number_g].scores # метрики этой группы

                for key, val in g.items():

                    if type(key) == int:
                        total += difficult2score(val)
                    else:
                        total += abs(subjects2hours[key]-val)*100
            
            return total

        


        self.bounds = np.array([[0, self.slots[i].size - 1] for i in range(1, 43)])
        
        def indexes2rows(indexes):
            return np.array([self.slots[i+1][int(j)] for i, j in enumerate(indexes)])


        self.eval = eval
        self.indexes2rows = indexes2rows





    def find_solution(self):

        func = lambda arr: self.eval(self.indexes2rows(arr))

        model = ga(func, 
                dimension = len(self.slots), 
                variable_type='int', 
                 variable_boundaries = self.bounds,
                 variable_type_mixed = None, 
                 function_timeout = 10,
                 algorithm_parameters={'max_num_iteration': 1000,
                                       'population_size':100,
                                       'mutation_probability':0.1,
                                       'elit_ratio': 0.05,
                                       'crossover_probability': 0.5,
                                       'parents_portion': 0.3,
                                       'crossover_type':'uniform',
                                       'selection_type': 'linear_ranking',
                                       'max_iteration_without_improv':None}
            )



        available_values = [np.arange(arr[0], arr[1]+1) for arr in self.bounds] #[self.slots[i+1] for i in range(42)]

        my_local_optimizer = lambda arr, score: Hill_Climbing_descent(function = func, 
            available_predictors_values=available_values, 
            random_counts_by_predictors = 200,
            max_function_evals = 500, 
            start_solution=arr )


        
        model.run(
            no_plot = False, 
            disable_progress_bar = False,
            disable_printing = False,

            set_function = None, 
            apply_function_to_parents = False, 
            start_generation = {'variables':None, 'scores': None},
            studEA = True,
            mutation_indexes = None,

            init_creator = None,
            init_oppositors = None,
            duplicates_oppositor = None,
            remove_duplicates_generation_step = None,

            revolution_oppositor = None,
            revolution_after_stagnation_step = None,
            revolution_part = 0.3,
            
            #population_initializer = Population_initializer(select_best_of = 10, local_optimization_step = 'after_select', local_optimizer = my_local_optimizer),
            
            stop_when_reached = None,
            callbacks = [],
            middle_callbacks = [],
            time_limit_secs = None, 
            save_last_generation_as = None,
            seed = None
            )


        self.model = model












    def save_results(self, file_name = 'output.xlsx'):

        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')


        conf = self.model.output_dict['variable']

        conf = self.indexes2rows(conf)

        vec = conf.astype(int)
        mask = vec >= 0
        
        slots = np.arange(vec.size)[mask] # реально используемые слоты
        df = self.combinations.iloc[vec[mask], :].copy() # вся нужная инфа про слот -- препод, предмет, аудитория, группа

        df_base = df.copy()

        df['slot'] = slots + 1
        df['group'] = [self.result['groups2names'][g] for g in df['group']]

        df.columns = ['Преподаватель', 'Предмет', 'Аудитория', 'Группа', 'Слот']

        df.to_excel(writer, sheet_name = 'Полные данные', index = False, startrow=3, startcol=3)



        GS, PS, TS = self.result['groups_schedule'], self.result['places_schedule'], self.result['teacher_schedule']
        
        names2group = {val:int(key) for key, val in self.result['groups2names'].items()}
        
        def slot2index(slot):
            s = slot - 1
            return (s%7, s//7 + 1)        
        tch = self.result['table_slots'].copy()
        tch.iloc[:, 1:] = '' # удаляю номера слотов
        gs = tch.copy()   


        for i in range(df.shape[0]):
            
            teacher, subject, place, group, slot = df.values[i,:]

            GS.iloc[names2group[group] - 1, int(slot)] = subject
            PS.iloc[int(place) - 1, int(slot)] = group
            TS.iloc[list(TS['Преподаватель']).index(teacher), int(slot)] = subject

            row, col = slot2index(slot)
            tch.iloc[row, col] = teacher
            gs.iloc[row, col] = group

        
        GS['Группа'] = [self.result['groups2names'][int(g)] for g in  GS['Группа'] ]
        

        GS.to_excel(writer, sheet_name = 'Группа-предмет', index = False, startrow=5, startcol=2)
        PS.to_excel(writer, sheet_name = 'Аудитория-группа', index = False, startrow=5, startcol=2)
        TS.to_excel(writer, sheet_name = 'Преподаватель-предмет', index = False, startrow=5, startcol=2)

        tch.to_excel(writer, sheet_name = 'Расписание преподавателей', index = False, startrow=5, startcol=2)
        gs.to_excel(writer, sheet_name = 'Расписание групп', index = False, startrow=5, startcol=2)





        # тут много копипаста, но что уж поделать
        # выводит метрики таблицами

        subjects2hours = self.result['subjects2hours']
        subjects2diffs = self.result['subjects2diffc']

        all_subjects = list(subjects2diffs.keys()) + [i for i in range(1, 7)]

        all_groups = self.result['all_groups']
        group_keys = {group.id: i for i, group in enumerate(all_groups)} # чтобы по айди группы брать номер группы в списке
        last_groups = [i for i, g in enumerate(all_groups) if g.last] # группы без детей - по ним метрики и считаем
        timeslot2day = np.zeros(len(self.slots), dtype = int) # соответствие между номером слота и днём
        for i in range(1, 7):
            timeslot2day[(i-1)*7:i*7] = i
        slot_indexes = np.arange(len(self.slots))


        df = df_base.loc[:, ['subject', 'group']].values

        for g in all_groups: # обнуляю счетчики
            g.set_subjects(all_subjects)


        for i, subj, group in zip(slot_indexes[mask], df[:,0], df[:,1]):

            dct = {
                    subj: 1, # +1 час по нужному предмету
                    timeslot2day[i]: subjects2diffs[subj] # сложность в этом дне
                }

            all_groups[group_keys[group]].add_score(dct, all_groups)



        difficults = []
        counts = []

        for number_g in last_groups:
    
            g = all_groups[number_g].scores # метрики этой группы
            name = self.result['groups2names'][all_groups[number_g].id]

            for key, val in g.items():

                if type(key) == int:
                    counts.append((name, key, val, difficult2score(val))) # сложности
                else:
                    difficults.append((name, key, val, subjects2hours[key], abs(subjects2hours[key]-val)*100)) # число часов 


        pd.DataFrame(counts, columns = ['Группа', 'День', 'Сложность', 'Штраф']).to_excel(writer, sheet_name = 'Сложности', index = False, startrow=5, startcol=2)
        pd.DataFrame(difficults, columns = ['Группа', 'Предмет', 'Число часов', 'Сколько должно быть', 'Штраф']).to_excel(writer, sheet_name = 'Часы', index = False, startrow=5, startcol=2)






        writer.save()

            







        




