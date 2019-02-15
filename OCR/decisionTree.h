/**
 * (C) Copyright 2018
 * 
 * Optical Character Recognition pe cifre scrise de mână
 */

#ifndef DECISIONTREE_H_
#define DECISIONTREE_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

bool same_class(const std::vector<std::vector<int>> &);

float get_entropy(const std::vector<std::vector<int>> &);

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> split(
    const std::vector<std::vector<int>> &, const int, const int);

std::pair<std::vector<int>, std::vector<int>> get_split_as_indexes(
    const std::vector<std::vector<int>> &, const int, const int);

float get_entropy_by_indexes(const std::vector<std::vector<int>> &,
                             const std::vector<int> &);

std::vector<int> random_dimensions(const int);

std::vector<int> compute_unique(const std::vector<std::vector<int>> &,
                                const int);

std::pair<int, int> find_best_split(const std::vector<std::vector<int>> &,
                                    const std::vector<int> &);

// structura unui nod din decision tree
// split_index = dimensiunea in functie de care se imparte
// split_value = valoarea in functie de care se imparte
// is_leaf si result sunt pentru cazul in care avem un nod frunza
class Node {
 protected:
    int split_index;
    int split_value;
    bool is_leaf;
    int result;
    std::shared_ptr<Node> left, right;

 public:
    Node();
    void make_decision_node(const int, const int);
    void make_leaf(const std::vector<std::vector<int>> &, const bool);
    void train(const std::vector<std::vector<int>> &);
    int predict(const std::vector<int> &) const;
};

#endif  // DECISIONTREE_H_
