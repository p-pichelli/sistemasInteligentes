import random
import numpy as np
import matplotlib.pyplot as plt

def algoritmoGenetico(distanciaCidades, populacao = 100, geracoes = 500, probCrossover = 0.87,
                      probMutacao = 0.05, nTorneio = 2):
    n = len(distanciaCidades)
    def inicializaPopulacao():
        return [random.sample(range(n), n) for _ in range(populacao)]
    def custoCaminho(caminho):
        return sum(distanciaCidades[caminho[i], caminho[i+1]] for i in range(n-1)) + distanciaCidades[caminho[-1], caminho[0]]
    def fitness(caminho, maxCusto, minCusto):
        custo = custoCaminho(caminho)
        return (maxCusto - custo) / (maxCusto - minCusto + 1e-6)
    def torneio(populacao, fitnesses, k=nTorneio):
        selecionados = random.sample(list(zip(populacao, fitnesses)), k)
        selecionados.sort(key=lambda x: x[1], reverse=True)
        return selecionados[0][0]
    def crossover_OX(pai1, pai2):
        n = len(pai1)
        filho = [None] * n
        corte1, corte2 = sorted(random.sample(range(n), 2))
        filho[corte1:corte2+1] = pai1[corte1:corte2+1]
        pos = (corte2 + 1) % n
        for cidade in pai2:
            if cidade not in filho:
                filho[pos] = cidade
                pos = (pos + 1) % n
        return filho
    def mutacao(caminho):
        i, j = random.sample(range(n), 2)
        caminho[i], caminho[j] = caminho[j], caminho[i]
        return caminho
    
    populacao_atual = inicializaPopulacao()
    melhor_caminho = None
    melhor_fitness = float('-inf')
    melhor_custo = 0
    historico_melhores = []
    custos_populacao = []
    for geracao in range(geracoes):
        custos = [custoCaminho(ind) for ind in populacao_atual]
        maxCusto = max(custos)
        minCusto = min(custos)
        populacao_atual.sort(key=lambda ind: fitness(ind, maxCusto, minCusto), reverse=True)
        fitnesses = [fitness(ind, maxCusto, minCusto) for ind in populacao_atual]
        if fitnesses[0] > melhor_fitness:
            melhor_custo = custoCaminho(populacao_atual[0])
            melhor_fitness = fitnesses[0]
            melhor_caminho = populacao_atual[0]
        historico_melhores.append((melhor_caminho, melhor_custo))
        num_elite = int(0.1 * populacao)
        selecionados = populacao_atual[:num_elite]
        while len(selecionados) < populacao:
            if random.random() < probCrossover:
                pai1 = torneio(populacao_atual, fitnesses)
                pai2 = torneio(populacao_atual, fitnesses)
                filho = crossover_OX(pai1, pai2)
                if random.random() < probMutacao:
                    filho = mutacao(filho)
                selecionados.append(filho)
            else:
                individuo = random.choice(selecionados)
                if random.random() < probMutacao:
                    individuo = mutacao(individuo)
                selecionados.append(individuo)
        populacao_atual = selecionados
        custos_geracao = [custoCaminho(ind) for ind in populacao_atual]
        custos_populacao.append(custos_geracao)
    return melhor_caminho, melhor_custo, historico_melhores, custos_populacao

if __name__ == "__main__":
    distanciaCidades = np.array([
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
    [20, 15, 0, 15, 14, 25, 81, 9, 23, 27],
    [21, 29, 15, 0, 4, 12, 92, 12, 25, 13],
    [16, 28, 14, 4, 0, 16, 94, 9, 20, 16],
    [31, 40, 25, 12, 16, 0, 95, 24, 36, 3],
    [100, 72, 81, 92, 94, 95, 0, 90, 101, 99],
    [12, 21, 9, 12, 9, 24, 90, 0, 15, 25],
    [4, 29, 23, 25, 20, 36, 101, 15, 0, 35],
    [31, 41, 27, 13, 16, 3, 99, 25, 35, 0]
])
    melhor_caminho, melhor_custo, historico, custos_populacao = algoritmoGenetico(distanciaCidades)
    print("Melhor caminho:", melhor_caminho)
    print("Custo do melhor caminho:", melhor_custo)
    for geracao, (caminho, custo) in enumerate(historico):
        print(f"Geração {geracao}: Caminho {caminho} com custo {custo}")

    # Plotagem do custo médio por geração
    media_por_geracao = [sum(custos)/len(custos) for custos in custos_populacao]
    plt.plot(range(len(media_por_geracao)), media_por_geracao, marker='o')
    plt.xlabel('Geração')
    plt.ylabel('Custo médio da população')
    plt.title('Custo médio por geração')
    plt.grid(True)
    plt.show()