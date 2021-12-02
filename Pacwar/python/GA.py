import os
import copy
import json
import random
import _PyPacwar
import numpy as np


class GA(object):
    def __init__(self, gene1, gene2, population_size=50, mutation_prob=0.01, turns=10, num_iters=100, eliminate_ratio=0.2,
                runID="v0"):
        self.gene1 = gene1
        self.gene2 = gene2
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.turns = turns
        self.num_iters = num_iters
        self.eliminate_ratio = eliminate_ratio
        self.runID = runID


    def get_scores(self, rounds, c1, c2):
        if c1 == c2:
            return [10, 10]

        species = [c1, c2]
        scores = [0, 0]
        winner = 0 if c1 > c2 else 1
        loser = abs(1 - winner)

        if species[loser] == 0:
            if rounds < 100:
                scores[winner], scores[loser] = 20, 0 
            elif 100 <= rounds <= 199:
                scores[winner], scores[loser] = 19, 1
            elif 200 <= rounds <= 299:
                scores[winner], scores[loser] = 18, 2
            elif 300 <= rounds <= 500:
                scores[winner], scores[loser] = 17, 3
        else:
            if species[winner] >= 10 * species[loser]:
                scores[winner], scores[loser] = 13, 7
            elif 3 * species[loser] <= species[winner] < 10 * species[loser]:
                scores[winner], scores[loser] = 12, 8
            elif 1.5 * scores[loser] <= species[winner] < 3 * species[loser]:
                scores[winner], scores[loser] = 11, 9
            else:
                scores[winner], scores[loser] = 10, 10

        return scores


    def evaluate(self, geneList1, geneList2, get_info=False):
        gene1_score = [0.0] * len(geneList1)
        gene2_score = [0.0] * len(geneList2)
        
        for i in range(len(geneList1)):
            for j in range(len(geneList2)):
                (rounds, c1, c2) = _PyPacwar.battle(geneList1[i], geneList2[j])
                
                scores = self.get_scores(rounds, c1, c2)
                gene1_score[i] += scores[0]
                gene2_score[j] += scores[1]

                if get_info:
                    print("The duel lasted %d rounds, gene 1 remains: %d, gene 2 remains: %d" % (rounds, c1, c2))
                    print("The final scores for geneList1[%d] vs geneList2[%d] are: %d : %d" % (i, j, scores[0], scores[1]))
        gene1_score = [i / len(geneList2) for i in gene1_score]
        gene2_score = [i / len(geneList1) for i in gene2_score]
        return gene1_score, gene2_score


    def crossover(self, gene1, gene2, k):
        '''
        gene1: A list of improving gene
        gene2: A list of comparing gene
        k: Point to crossover, two genes will exchange the parts after this given pivot
        '''
        gene1, gene2 = gene1[:k] + gene2[k:], gene2[:k] + gene1[k:]
        return gene1, gene2


    def mutation(self, gene, mutation_prob):
        for i in range(len(gene)):
            if np.random.rand() < mutation_prob:
                gene[i] = np.random.choice(4)   # 0, 1, 2, 3
        return gene


    def evolusion(self, geneList, scoreList, mutation_prob):
        # scoreList: a list scores of geneList
        pop_size = len(geneList)
        scores_gene = zip(scoreList, geneList)

        # Sort the scores by descend order
        scores_gene = sorted(scores_gene, key=lambda x: x[0], reverse=True)
        elimate_idx = int(pop_size * self.eliminate_ratio)
        survive_idx = int(pop_size * (1 - self.eliminate_ratio))
        
        # Duplicate the gene with the highest score, remove the gene with lowest score
        scores_gene = scores_gene[:survive_idx] + scores_gene[:elimate_idx]
        
        # Unzip scores_gene
        prob, gene = zip(*scores_gene)
        gene = list(gene)
        prob = list(prob)

        # Normalize
        prob = [i / float(sum(prob)) for i in prob]

        for i in range(0, pop_size, 2):
            if i + 1 >= len(gene):
                break
            k = np.random.choice(len(gene[i]))
            gene[i], gene[i+1] = self.crossover(gene[i], gene[i+1], k)
            gene[i], gene[i+1] = self.mutation(gene[i], mutation_prob), self.mutation(gene[i+1], mutation_prob)

        return gene


    def selection(self, geneList, scoreList):
        pop_size = len(geneList)

        scoreSum = float(sum(scoreList))
        prob = [(score / scoreSum) for score in scoreList]

        new_gene_idx = np.random.choice(pop_size, size=pop_size, p=prob)
        new_gene = [geneList[i] for i in new_gene_idx]
        new_score = [scoreList[i] for i in new_gene_idx]

        return new_score, new_gene


    def train(self):
        gene1, gene2 = self.gene1, self.gene2
        mutation_prob = self.mutation_prob

        for t in range(self.turns):
            for i in range(1, self.num_iters+1):
                # Fixed gene2 and only improve gene1, then fix gene1 searching gene2
                score1, score2 = self.evaluate(gene1, gene2)
                score1, gene1 = self.selection(gene1, score1)
                gene1 = self.evolusion(gene1, score1, mutation_prob)

                avg_score1 = "{:.2f}".format(sum(score1) / float(len(score1)))
                avg_score2 = "{:.2f}".format(sum(score2) / float(len(score2)))
                print(f"Turn: {t}, Step: {i}, Avg Score of Gene1: {avg_score1}, Gene2: {avg_score2}")

            self.save_turn(gene1, gene2, score1, score2, t, runID)
            print(f"\nGene Saved!")

            mean_score1 = "{:.2f}".format(np.mean(score1))
            mean_score2 = "{:.2f}".format(np.mean(score2))
            print(f"\nTurn: {t}, Gene1 score: {mean_score1}, Gene2 score: {mean_score2}")

            if(mean_score1 > mean_score2):
                # Pick top 10 to append to the new fixed gene 2
                new_gene2, new_score2 = self.topk(gene1, score1, k=10)
                gene2 += new_gene2
                score2 += new_score2
            gene2, score2 = self.topk(gene2, score2, k=self.population_size)
            print(f"\nTurn {t} completed, The poputation of the new gene 2 {len(gene2)}")


            top_gene, top_score = self.topk(gene1, score1, k=1)
            top_gene = "".join([str(i) for i in top_gene[0]])
            top_score = "{:.2f}".format(top_score[0])
            print(f"Gene1=> Best Gene: {top_gene}, Score: {top_score}")

            top_gene, top_score = self.topk(gene2, score2, k=1)
            top_gene = "".join([str(i) for i in top_gene[0]])
            top_score = "{:.2f}".format(top_score[0])
            print(f"Gene2=> Best Gene: {top_gene}, Score: {top_score}")
            gene1 = copy.deepcopy(gene2)

        top_gene, top_score = self.topk(gene1, score1, k=1)
        top_gene = "".join([str(i) for i in top_gene[0]])
        top_score = "{:.2f}".format(top_score[0])
        print(f"Gene1=> Best Gene: {top_gene}, Score: {top_score}")

        top_gene, top_score = self.topk(gene2, score2, k=1)
        top_gene = "".join([str(i) for i in top_gene[0]])
        top_score = "{:.2f}".format(top_score[0])
        print(f"Gene2=> Best Gene: {top_gene}, Score: {top_score}")

        return gene1, gene2


    def topk(self, geneList, scoreList, k=1):
        score_gene = zip(scoreList, geneList)
        score_gene = sorted(score_gene, key=lambda x: x[0], reverse=True)
        score, gene = zip(*score_gene)
        gene = list(gene)
        score = list(score)
        return gene[:k], score[:k]


    def save_turn(self, gene1, gene2, score1, score2, turn, runID):
        gene1_dict, gene2_dict = {}, {}

        for i in range(len(gene1)):
            gene_str = "".join([str(k) for k in gene1[i]])
            gene1_dict[gene_str] = score1[i]
        for i in range(len(gene2)):
            gene_str = "".join([str(k) for k in gene2[i]])
            gene2_dict[gene_str] = score2[i]

        gene1_json_str, gene2_json_str = json.dumps(gene1_dict), json.dumps(gene2_dict)
        if not os.path.exists("./Genes"):
            os.mkdir("./Genes")
        with open(f"Genes/{runID}_gene1_{turn}.json", "w") as f:
            json.dump(gene1_dict, f)
        with open(f"Genes/{runID}_gene2_{turn}.json", "w") as f:
            json.dump(gene2_dict, f)


if __name__ == '__main__':
    runID = "v6"
    init_population = 100
    np.random.seed(np.random.randint(1000))

    gene1 = list()
    #for i in range(init_population):
    #    gene1.append(np.random.randint(low=0, high=4, size=50).tolist())

    # Import gene that we want to improve
    gene1_string = "03000000021001033333322121231111221231131233133131"
    for i in range(init_population):
        gene_temp = list()
        for j in gene1_string:
            gene_temp.append(int(j))
        gene1.append(gene_temp)
    gene2 = [[3 if i == 0 else 1 for _ in range(50)] for i in range(2)]

    gaParwar = GA(gene1, gene2, population_size=100, mutation_prob=0.001, turns=100, num_iters=50, eliminate_ratio=0.2, runID=runID)
    gaParwar.train()
