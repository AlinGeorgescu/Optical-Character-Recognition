
build:
	g++ -o randomForest.o -c randomForest.cpp --std=c++11 -O2 -g
	g++ -o decisionTree.o -c decisionTree.cpp --std=c++11 -O2 -g
	g++ -o ex main.cpp randomForest.o decisionTree.o --std=c++11 -O2 -g
	make -C resources/ build


.PHONY: clean

clean:
	rm randomForest.o
	rm decisionTree.o
	rm ex
	make -C resources/ clean
