

import numpy as np
import pandas as pd


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

        # перевод сложности в метрику
        def difficult2score(value):
            if value > 20: return 50*(value-20)
            return value



        def eval(vector):

            nonlocal all_groups

            df = needed_df[vector.astype(int),:]

            for g in all_groups: # обнуляю счетчики
                g.set_subjects(all_subjects)


            for i, (subj, group) in enumerate(zip(df[:,0], df[:,1])):

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

        


        self.bounds = np.array([[0, self.slots[i].size] for i in range(1, 43)])
        
        def indexes2rows(indexes):
            return np.array([self.slots[i][j] for i, j in enumerate(indexes)])


        self.eval = eval
        self.indexes2rows = indexes2rows





    def find_solution(self):

        pass

    def save_results(self):

        pass
        




