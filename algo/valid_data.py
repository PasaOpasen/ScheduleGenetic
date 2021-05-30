
import itertools

import numpy as np
import pandas as pd



def get_all_combinations_df(result):
    """
    выдает все возможные допустимые комбинации препод-предмет-место-группа

    """

    # дальше я всё делаю не прям в один шаг, чтоб не расходовать дико много памяти

    # все комбинации учителей и их предметов
    teachers_subjects = sum([list(itertools.product([key], val)) for key, val in result['teachers'].items()], [])

    # все комбинации групп и аудиторий, но чтобы аудитория вмещала группу
    places_groups = sum([ [(place, group) for group, count2 in result['groups2counts'].items() if count1 >= count2] for place, count1 in result['places2counts'].items()], [])


    teachers_subjects_places_groups = list(itertools.product(teachers_subjects, places_groups))

    def connect(place, subject):
        pl = result['places2contain'][place]
        su = result['subjects2reqs'][subject]

        return su.issubset(pl) # место должно подходить под требование урока, но наоборот не обязательно


    teachers_subjects_places_groups = [(teacher, subject, place, group) for (teacher, subject), (place, group) in teachers_subjects_places_groups if connect(place, subject)]

    return pd.DataFrame(teachers_subjects_places_groups, columns = ['teacher', 'subject', 'place', 'group'])


def get_ways_for_timeslots(result):
    """
    возвращает все допустимые комбинации вообще и номера разрешенных комбинаций под каждый слот
    """

    all_combs = get_all_combinations_df(result)

    # all_combs = all_combs[all_combs['group'] < 4]

    slots = dict.fromkeys(range(1, result['teacher_schedule'].shape[1]))

    all_rows = np.arange(all_combs.shape[0])

    
    def get_keys_with_elem(dct, elem):
        return [key for key, val in dct.items() if elem in val]


    for slot in list(slots.keys()):
        
        group_ex = get_keys_with_elem(result['groups_exceptions'], slot)
        teacher_ex = get_keys_with_elem(result['teacher_exceptions'], slot)
        place_ex = get_keys_with_elem(result['places_exceptions'], slot)

        slots[slot] = all_rows[~(all_combs['teacher'].isin(teacher_ex) | all_combs['place'].isin(place_ex) | all_combs['group'].isin(group_ex))]
    
    return all_combs, {s: np.concatenate([np.full(5, -1), arr]) for s, arr in slots.items()} # добавляю таким образом возможность ничего не ставить в слоте, пустая пара



if __name__ == '__main__':
    
    all_combs, slots = get_ways_for_timeslots(result)









