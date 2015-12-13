//This is a C++ implementation for timing comparisons.
#include <random>
#include <stdio.h>
#include <time.h>
#include <limits>

const int WIDTH = 101;
const int HEIGHT = 101;
float workspace[WIDTH][HEIGHT];
float costs[WIDTH][HEIGHT];
const int START_X = 50;
const int START_Y = 50;
#define GET_INFINITY (std::numeric_limits<float>::infinity())

inline float min2(float a, float b){
	return a < b ? a : b;
}

inline float min4(float a, float b, float c, float d) {
	float p = min2(a, b);
	float q = min2(c, d);
	return min2(p, q);
}

template<typename T1, typename T2>
int automaton(T1 workspace, T2 costs, float epsilon) {
	int writes = 0;
	const float inf = GET_INFINITY;
	for(int x = 0; x < WIDTH; x++) {
		for(int y = 0; y < HEIGHT; y++) {
			float current = workspace[x][y];
			float tmp = min4(
				x+1 < WIDTH ? workspace[x+1][y]+0.5*costs[x+1][y] : inf,
				x-1 >= 0 ? workspace[x-1][y]+0.5*costs[x-1][y] : inf,
				y+1 < HEIGHT ? workspace[x][y+1]+0.5*costs[x][y+1] : inf,
				y-1 >= 0 ? workspace[x][y-1]+0.5*costs[x][y-1] : inf);
			tmp += 0.5*costs[x][y];
			if(tmp+epsilon < current) {
				workspace[x][y] = tmp;
				writes++;
			}			
		}
	}
	return writes;
}

//returns the epsilon
float prepare() {
	const float inf = GET_INFINITY;
	float epsilon = inf;
	std::mt19937 engine{};
	std::uniform_real_distribution<> gen{0.1, 1.0};
	for(int x = 0; x < WIDTH; x++) {
		for(int y = 0; y < HEIGHT; y++) {
			costs[x][y] = gen(engine);
			epsilon  = min2(epsilon, costs[x][y]);
			workspace[x][y] = inf;
		}
	}
	workspace[START_X][START_Y] = 0.0f;
	return epsilon/2.0;
}

template<typename T1, typename T2>
int mainloop(T1 workspace, T2 costs, float epsilon) {
	int writes_total = 0;
	int writes = 0;
	do {
		writes = automaton(workspace, costs, epsilon);
		writes_total += writes;
	} while(writes);
	return writes_total;
}

void main() {
	float epsilon = prepare();
	auto start = clock();
	int writes = mainloop(workspace, costs, epsilon);
	auto end = clock();
	double secs = (float)(end-start)/CLOCKS_PER_SEC;
	printf("Total time: %f\n", secs);
	printf("Writes: %i\n", writes);
}
