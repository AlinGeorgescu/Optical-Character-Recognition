/**
 * (C) Copyright 2018
 * 
 * Optical Character Recognition pe cifre scrise de mână
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>

#include "decisionTree.h"


using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;
using std::find;
using std::shared_ptr;
using std::mt19937;
using std::random_device;
using std::uniform_int_distribution;
// structura unui nod din decision tree
// splitIndex = dimensiunea in functie de care se imparte
// split_value = valoarea in functie de care se imparte
// is_leaf si result sunt pentru cazul in care avem un nod frunza
Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}

void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    // Seteaza nodul ca fiind de tip frunza (modificati is_leaf si result)
    // is_single_class = true -> toate testele au aceeasi clasa (acela e result)
    // is_single_class = false -> se alege clasa care apare cel mai des
    this->is_leaf = true;
    if (is_single_class) {
        this->result = samples[0][0];
        return;
    }
    vector<int> freq(10, 0);
    int max = 0;
    for (int i = 0; i < samples.size(); ++i) {
        ++freq[samples[i][0]];
    }
    for (int i = 0; i < 10; ++i) {
        if (freq[i] > max) {
            max = freq[i];
            this->result = i;
        }
    }
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    // Intoarce cea mai buna dimensiune si valoare de split dintre testele
    // primite. Prin cel mai bun split (dimensiune si valoare)
    // ne referim la split-ul care maximizeaza IG
    // pair-ul intors este format din (split_index, split_value)
    int splitIndex = -1, splitValue = -1;
    float max = -1.0;
    float HL, HR, entropyChildren, IG;
    float entropyTarget = get_entropy(samples);

    for (int i = 0; i < dimensions.size(); ++i) {
        int sum = 0, prod = 1;
        int index = dimensions[i];
        vector<int> col = compute_unique(samples, index);

        for (int j = 0; j < col.size(); ++j) {
            sum += col[j];
            prod *= col[j];
        }

        int mediaA = sum / col.size();
        int mediaG = pow(prod, 1 / col.size());
        auto pair1 = split(samples, index, mediaA);
        auto pair2 = split(samples, index, mediaG);
        int nl1 = pair1.first.size();
        int nr1 = pair1.second.size();
        int nl2 = pair2.first.size();
        int nr2 = pair2.second.size();

        if (abs(nl1 - nr1) < abs(nl2 - nr2)) {
            if (nl1 && nr1) {
                HL = get_entropy(pair1.first);
                HR = get_entropy(pair1.second);
                entropyChildren = (nl1 * HL + nr1 * HR) / (nl1 + nr1);
                IG = entropyTarget - entropyChildren;

                if (IG > max) {
                    max = IG;
                    splitIndex = index;
                    splitValue = mediaA;
                }
            } else if (nl2 && nr2) {
                HL = get_entropy(pair2.first);
                HR = get_entropy(pair2.second);
                entropyChildren = (nl2 * HL + nr2 * HR) / (nl2 + nr2);
                IG = entropyTarget - entropyChildren;

                if (IG > max) {
                    max = IG;
                    splitIndex = index;
                    splitValue = mediaG;
                }
            }
        } else {
            if (nl2 && nr2) {
                HL = get_entropy(pair2.first);
                HR = get_entropy(pair2.second);
                entropyChildren = (nl2 * HL + nr2 * HR) / (nl2 + nr2);
                IG = entropyTarget - entropyChildren;

                if (IG > max) {
                    max = IG;
                    splitIndex = index;
                    splitValue = mediaG;
                }
            } else if (nl1 && nr1) {
                HL = get_entropy(pair2.first);
                HR = get_entropy(pair2.second);
                entropyChildren = (nl1 * HL + nr1 * HR) / (nl1 + nr1);
                IG = entropyTarget - entropyChildren;

                if (IG > max) {
                    max = IG;
                    splitIndex = index;
                    splitValue = mediaA;
                }
            }
        }
    }

    if (max < 0)
        return pair<int, int>(-1, -1);

    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    // Antreneaza nodul curent si copii sai, daca e nevoie
    // 1) verifica daca toate testele primite au aceeasi clasa (raspuns)
    // Daca da, acest nod devine frunza, altfel continua algoritmul.
    // 2) Daca nu exista niciun split valid, acest nod devine frunza. Altfel,
    // ia cel mai bun split si continua recursiv
    bool is_single_class = same_class(samples);

    if (is_single_class) {
        this->make_leaf(samples, is_single_class);
    } else {
        vector<int> dim = random_dimensions(samples[0].size());
        auto p = find_best_split(samples, dim);

        if (p.first == -1 && p.second == -1){
            this->make_leaf(samples, is_single_class);
        } else {
            this->make_decision_node(p.first, p.second);

            vector<vector<int>> child1, child2;
            auto q = split(samples, p.first, p.second);
            child1 = q.first;
            child2 = q.second;

            left = make_shared<Node>();
            right = make_shared<Node>();
            left->train(child1);
            right->train(child2);
        }
    }
}

int Node::predict(const vector<int> &image) const {
    // Intoarce rezultatul prezis de catre decision tree
    if (this->is_leaf)
        return this->result;

    if (image[split_index - 1] <= split_value)
        return left->predict(image);
    else
        return right->predict(image);
}

bool same_class(const vector<vector<int>> &samples) {
    // Verifica daca testele primite ca argument au toate aceeasi
    // clasa(rezultat). Este folosit in train pentru a determina daca
    // mai are rost sa caute split-uri
    for (int i = 1; i < samples.size(); ++i) {
        if (samples[i][0] != samples[0][0])
            return false;
    }

    return true;
}

float get_entropy(const vector<vector<int>> &samples) {
    // Intoarce entropia testelor primite
    assert(!samples.empty());
    int size = samples.size();
    vector<int> indexes(size);

    // Am optimizat adăugarea în vector, deoarece push_back este mai lent
    for (int i = 0; i < size; i++)
        indexes[i] = i;

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    // Intoarce entropia subsetului din setul de teste total(samples)
    // Cu conditia ca subsetul sa contina testele ale caror indecsi se gasesc in
    // vectorul index (Se considera doar liniile din vectorul index)
    vector<int> freq(10, 0);
    float entropy = 0.0;
    float total = index.size();

    for (int i = 0; i < total; ++i) {
        ++freq[ samples[ index[i] ][ 0 ] ];
    }

    for (int i = 0; i < 10; ++i) {
        if (freq[i]) {
            float Pi = freq[i] / total;
            entropy -= Pi * log2(Pi);
        }
    }

    return entropy;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    // Intoarce toate valorile (se elimina duplicatele)
    // care apar in setul de teste, pe coloana col
    vector<int> uniqueValues;
    int mark[256] = { 0 };

    for (int i = 0; i < samples.size(); ++i) {
        if (!mark[ samples[i][col] ]) {
            uniqueValues.push_back(samples[i][col]);
            ++mark[ samples[i][col] ];
        }
    }

    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce cele 2 subseturi de teste obtinute in urma separarii
    // In functie de split_index si split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce indecsii sample-urilor din cele 2 subseturi obtinute in urma
    // separarii in functie de split_index si split_value
    vector<int> left, right;

    for (int i = 0; i < samples.size(); ++i) {
        if (samples[i][split_index] <= split_value)
            left.push_back(i);
        else
            right.push_back(i);
    }

    return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
    // Intoarce sqrt(size) dimensiuni diferite pe care sa caute splitul maxim
    // Precizare: Dimensiunile gasite sunt > 0 si < size
    int generated;
    int count = 0;
    int dim = floor(sqrt(size));
    vector<int> rez(dim);
    int mark[size] = { 0 };
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(1, size - 1);

    while (count < dim) {
        generated = dist(mt) % size;
        if (!mark[generated]) {
            ++mark[generated];
            rez[count++] = generated;
        }
    }

    return rez;
}
