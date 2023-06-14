import numpy as np

def target_function(x):
    return np.sin(10*np.pi*x)*x + 2.0

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, elitism_count):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count

    def init_population(self):
        return np.random.uniform(low=-1.0, high=2.0, size=self.population_size)

    def calculate_fitness(self, population):
        return target_function(population)

    def perform_selection(self, population, fitness):
        return population[np.argsort(fitness)][::-1]

    def perform_crossover(self, population):
        for i in range(self.elitism_count, self.population_size-1, 2):
            if np.random.random() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.population_size-1)
                temp = population[i].copy()
                population[i] = population[i+1].copy()
                population[i+1] = temp.copy()
        return population

    def perform_mutation(self, population):
        for i in range(self.elitism_count, self.population_size):
            if np.random.random() < self.mutation_rate:
                population[i] = np.random.uniform(low=-1.0, high=2.0)
        return population

    def run(self, max_generations):
        population = self.init_population()
        for generation in range(max_generations):
            fitness = self.calculate_fitness(population)
            population = self.perform_selection(population, fitness)
            population = self.perform_crossover(population)
            population = self.perform_mutation(population)

            if generation % 20 == 0:
                best_fit = np.max(fitness)
                print(f"Generation: {generation}, Best Fit: {best_fit}")

        return population

if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=100, mutation_rate=0.01, crossover_rate=0.9, elitism_count=2)
    ga.run(max_generations=200)
