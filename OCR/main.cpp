/**
 * (C) Copyright 2018
 * 
 * Optical Character Recognition pe cifre scrise de mână
 */

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "randomForest.h"

using std::pair;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::cerr;
using std::cout;
using std::stringstream;
using std::flush;

pair<vector<vector<int>>, vector<vector<int>>> read() {
    vector<vector<int>> train, test;
    ifstream f("mnist_train.csv");
    if (!f) {
        cerr << "Error opening file\n";
        exit(1);
    }

    vector<int> l;
    string line;
    int nr;
    while (getline(f, line)) {
        stringstream parser(line);
        l.clear();
        while (parser >> nr) {
            l.push_back(nr);
            if (parser.peek() == ',') parser.ignore();
        }
        train.push_back(l);
    }

    f.close();

    ifstream g("mnist_test.csv");
    if (!g) {
        cerr << "Error opening file\n";
        exit(1);
    }

    while (getline(g, line)) {
        stringstream parser(line);
        l.clear();
        while (parser >> nr) {
            l.push_back(nr);
            if (parser.peek() == ',') {
                parser.ignore();
            }
        }
        test.push_back(l);
    }

    g.close();
    return pair<vector<vector<int>>, vector<vector<int>>>(train, test);
}

int main() {
    int seed = time(0);
    srand(seed);

    pair<vector<vector<int>>, vector<vector<int>>> input = read();
    vector<vector<int>> train = input.first, test = input.second;

    RandomForest forest(10, train);
    forest.build();

    int correct = 0, ans;
    for (const auto &it : test) {
        vector<int> vec;
        vec.reserve(it.size());
        copy(it.begin() + 1, it.end(), back_inserter(vec));
        ans = forest.predict(vec);
        if (ans == it[0]) correct++;
    }

    float precision =
        static_cast<float>(correct) / static_cast<float>(test.size()) * 100;
    cerr << "Precision: " << precision << "%\n" << flush;

    if (precision > 85)
        cout << "30";
    else if (precision > 55)
        cout << "20";
    else if (precision > 25)
        cout << "10";
    else
        cout << "0";

    return 0;
}
