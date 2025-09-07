import random
import numpy as np
import matplotlib.pyplot as plt

def algoritmoGenetico(distanciaCidades, populacao = 200, geracoes = 500, probCrossover = 0.87,
                      probMutacao = 0.15, nTorneio = 2):
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
    [0, 12, 19, 23, 31, 18, 21, 17, 25, 14, 28, 22, 16, 27, 20, 24, 26, 15, 29, 18],
    [12, 0, 15, 28, 14, 26, 13, 22, 30, 19, 17, 21, 23, 25, 18, 20, 27, 16, 24, 29],
    [19, 15, 0, 16, 27, 24, 20, 25, 18, 22, 21, 19, 28, 17, 23, 26, 14, 29, 20, 21],
    [23, 28, 16, 0, 22, 17, 29, 24, 15, 27, 18, 20, 25, 21, 19, 30, 16, 23, 28, 24],
    [31, 14, 27, 22, 0, 13, 18, 16, 29, 20, 15, 17, 21, 24, 22, 19, 25, 28, 14, 23],
    [18, 26, 24, 17, 13, 0, 11, 20, 15, 22, 19, 14, 16, 18, 21, 23, 17, 25, 20, 27],
    [21, 13, 20, 29, 18, 11, 0, 12, 23, 17, 16, 19, 14, 20, 18, 15, 22, 21, 24, 16],
    [17, 22, 25, 24, 16, 20, 12, 0, 19, 14, 21, 18, 23, 15, 17, 20, 13, 16, 22, 25],
    [25, 30, 18, 15, 29, 15, 23, 19, 0, 21, 24, 17, 20, 22, 16, 18, 27, 14, 19, 20],
    [14, 19, 22, 27, 20, 22, 17, 14, 21, 0, 13, 16, 18, 15, 19, 23, 20, 25, 17, 21],
    [28, 17, 21, 18, 15, 19, 16, 21, 24, 13, 0, 22, 20, 18, 23, 17, 14, 19, 25, 16],
    [22, 21, 19, 20, 17, 14, 19, 18, 17, 16, 22, 0, 15, 20, 18, 21, 23, 17, 14, 19],
    [16, 23, 28, 25, 21, 16, 14, 23, 20, 18, 20, 15, 0, 19, 17, 22, 21, 24, 18, 20],
    [27, 25, 17, 21, 24, 18, 20, 15, 22, 15, 18, 20, 19, 0, 21, 17, 23, 16, 20, 22],
    [20, 18, 23, 19, 22, 21, 18, 17, 16, 19, 23, 18, 17, 21, 0, 15, 20, 22, 19, 16],
    [24, 20, 26, 30, 19, 23, 15, 20, 18, 23, 17, 21, 22, 17, 15, 0, 14, 19, 21, 18],
    [26, 27, 14, 16, 25, 17, 22, 13, 27, 20, 14, 23, 21, 23, 20, 14, 0, 18, 15, 19],
    [15, 16, 29, 23, 28, 25, 21, 16, 14, 25, 19, 17, 24, 16, 22, 19, 18, 0, 21, 20],
    [29, 24, 20, 28, 14, 20, 24, 22, 19, 17, 25, 14, 18, 20, 19, 21, 15, 21, 0, 23],
    [18, 29, 21, 24, 23, 27, 16, 25, 20, 21, 16, 19, 20, 22, 16, 18, 19, 20, 23, 0]
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