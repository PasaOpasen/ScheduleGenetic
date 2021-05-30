
from typing import List


class Group:
    """
    класс группы, позволяющий распространять какие-то метрики на саму группу и её потомков

    то есть если к потоку из группы 1 и группы 2 прибавляем 5,
    эта 5 прибавится и к потоку и ко группа и к их подгруппам и т. д.

    """

    def __init__(self, my_id: int, childs:List[int] = None):
        
        self.id = my_id
        self.last = childs is None
        if not self.last: self.childs = set(childs)
        #self.score = 0

    def is_child(self, group):

        return group.id in self.childs

    def set_subjects(self, subjects):

        self.scores = {s:0 for s in subjects}


    def add_score(self, scores, all_groups:List):

        for key, score in scores.items():
            self.scores[key] += score

        if not self.last:
            for group in all_groups:

                if self.is_child(group):

                    group.add_score(scores, all_groups)





if __name__ == '__main__':

    g1 = Group(1, [2,3])
    g4 = Group(4, [5,6])
    g2 = Group(2)
    g3 = Group(3)
    g5 = Group(5)
    g6 = Group(6)

    all_groups = [g1,g2,g3,g4,g5,g6]

    for g in all_groups:
        g.set_subjects(['1', '2', 4])

    g1.add_score( {'1': 10}, all_groups)
    g4.add_score({'1': 1, 4: 2}, all_groups)

    g5.add_score({'1': 2}, all_groups)

    print([g.scores for g in all_groups])










