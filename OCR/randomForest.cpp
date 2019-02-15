/**
 * (C) Copyright 2018
 * 
 * Optical Character Recognition pe cifre scrise de mână
 */

#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>

#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;
using std::max_element;
using std::distance;
using std::random_device;
using std::uniform_int_distribution;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    vector<vector<int>> ret;
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(0, samples.size());

    int i = 0;
    vector<int> mark(samples.size(), 0);

    while (i < num_to_return) {
        int index = dist(mt) % samples.size();

        if (!mark[index]) {
            ret.push_back(samples[index]);
            ++i;
        }
        ++mark[index];
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // Aloca pentru fiecare Tree cate n / num_trees
    // Unde n e numarul total de teste de training
    // Apoi antreneaza fiecare tree cu testele alese
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        random_samples = get_random_samples(images, data_size);

        // Construieste un Tree nou si il antreneaza
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // Va intoarce cea mai probabila prezicere pentru testul din argument
    // se va interoga fiecare Tree si se va considera raspunsul final ca
    // fiind cel majoritar
    int max = 0;
    int pos;
    int freq[10] = {0};
    vector<Node>::iterator it;
    for (it = trees.begin(); it != trees.end(); ++it) {
        ++freq[ it->predict(image) ];
    }

    for (int i = 0; i < 10; ++i) {
        if (freq[i] > max) {
            max = freq[i];
            pos = i;
        }
    }

    return pos;
}
