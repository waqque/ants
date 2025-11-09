import numpy as np
import matplotlib.pyplot as plt

def read_graph(filename): # считывание матрциы из файла 
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            s, t, w = map(int, line.split()) #source target weight 
            edges.append((s, t, w))
    
    #  количество вершин опреденение
    n = max(max(s, t) for s, t, w in edges) + 1
    
    #  матрица расстояний создание
    dist = np.full((n, n), 999999)
    np.fill_diagonal(dist, 0)  # расстояние до себя = 0
    
    # матрица расстояний заполнение
    for s, t, w in edges:
        dist[s][t] = w
        dist[t][s] = w  # граф неориентированный
    
    return dist, n

def ant_algorithm(distances, ants=20, iterations=100, decay=0.1, alpha=1.0, beta=2.0, random_seed=42):
    np.random.seed(random_seed)
    
    n = len(distances)
    
    # инициализация феромонов
    pheromone = np.ones((n, n)) / n
    np.fill_diagonal(pheromone, 0)  # нет феромонов на диагонали
    
    best_path = None
    best_dist = float('inf')
    dist_history = []
    pheromone_history = []
    path_history = []
    
    for it in range(iterations):
        paths = []
        
        # каждый муравей строит путь
        for ant in range(ants):
            path = [0]  # начинает с вершины 0
            visited = set([0])
            
            #  путь строится пока не посетим все вершины
            while len(path) < n:
                current = path[-1]
                probs = []
                
                # вероятность для всех доступных вершин
                for next_node in range(n):
                    if (next_node not in visited and 
                        distances[current][next_node] < 999999):
                        # выбор следующей вершины ориентируясь на феромоны
                        pheromone_component = pheromone[current][next_node] ** alpha
                        heuristic_component = (1.0 / distances[current][next_node]) ** beta
                        probability = pheromone_component * heuristic_component
                        probs.append((next_node, probability))
                
                # если нет доступных вершин, прерываем построение
                if not probs:
                    break
                    
                # следующая вершину по вероятностям
                nodes, probabilities = zip(*probs)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()  # нормализуем
                
                next_node = np.random.choice(nodes, p=probabilities)
                path.append(next_node)
                visited.add(next_node)
            
            # вычислет длину пути только если посетили все вершины
            if len(path) == n:
                total_distance = 0
                for i in range(len(path) - 1):
                    total_distance += distances[path[i]][path[i + 1]]
                paths.append((path, total_distance))
                
                # обновление лучшего пути
                if total_distance < best_dist:
                    best_path = path
                    best_dist = total_distance
        #сохранение лучшего пути
        dist_history.append(best_dist)
        pheromone_history.append(np.mean(pheromone))
        if best_path is not None:
            path_history.append(best_path.copy())
        
        # испарение феромонлв
        pheromone *= (1 - decay)
        
        # обновл феромоны на лучшем пути
        if best_path is not None and len(best_path) == n:
            for i in range(len(best_path) - 1):
                pheromone[best_path[i]][best_path[i + 1]] += 1.0 / best_dist
                pheromone[best_path[i + 1]][best_path[i]] += 1.0 / best_dist  # симметрично
    
    return best_path, best_dist, dist_history, pheromone_history, path_history

def multiple_runs(filename, num_runs=3, ants=20, iterations=100): #запускает муравьиный алг несколкьо раз для лучшего рез-та количество пробегов = 3
    distances, n = read_graph(filename)
    best_overall_path = None
    best_overall_dist = float('inf')
    all_results = []
    
    print(f"Запуск муравьиного алгоритма {num_runs} раз:")
    
    for run in range(num_runs):
        path, dist, dist_hist, pheromone_hist, path_hist = ant_algorithm(
            distances, 
            ants=ants, 
            iterations=iterations, 
            random_seed=run 
        )
        
        all_results.append((path, dist))
        
        if path is not None and dist < best_overall_dist:
            best_overall_path = path
            best_overall_dist = dist
        
        print(f"Запуск {run + 1}:длина = {dist}, путь = {path}")
    print(f"Лучший результат:длина = {best_overall_dist}")
    print(f"Лучший путь: {best_overall_path}")
    
    return best_overall_path, best_overall_dist, all_results

# основная программа
if __name__ == "__main__":
    distances, n = read_graph("20.txt")
    print(f"В графе: {n} вершин")
    
    print("\n1.Один запуск:")
    path, dist, dist_hist, pheromone_hist, path_hist = ant_algorithm(
        distances, 
        ants=20, 
        iterations=100, 
        random_seed=42
    )
    
    print(f"Лучший путь: {path}")
    print(f"Длина: {dist}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pheromone_hist)
    plt.title('Средний уровень феромонов по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('Феромоны')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dist_hist)
    plt.title('Длина лучшего пути по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('Длина')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n2.Несколько запусков для поиска лучшего решения(3):")
    best_path, best_dist, all_results = multiple_runs(
        "20.txt", 
        num_runs=3, 
        ants=20, 
        iterations=100
    )