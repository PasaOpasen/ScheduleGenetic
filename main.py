

from algo import read_all_data, get_ways_for_timeslots, Optimizator












if __name__ == '__main__':

    for i in range(30):
    
        result = read_all_data('./data/Diss.xlsx') # перевод данные из эксель в словарь с нужными штуками
    
        all_combs, slots = get_ways_for_timeslots(result) # создает всевозможные допустимые комбинации для каждого слота
    
        opt = Optimizator(result, all_combs, slots) # инициализируем алгоритм
    
        opt.set_evaluators() # кешируем многие штуки, нужные для поиска решения
    
        opt.find_solution(i) # поиск решения
        
        
        opt.save_results()



























