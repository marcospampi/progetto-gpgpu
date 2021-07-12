#include <iostream>
#include <chrono>
#include <climits>
#include <ctime>
#include <cstdlib>
#define TEST_SIZE 4096
struct R { int min{INT_MIN}, max{INT_MAX}; };
R minmax( int * input, int len ) {
	R r;
	for ( int i = 0; i < len; ++i ) {
		r.min = r.min < input[i] ? r.min : input[i];
		r.max = r.max > input[i] ? r.max : input[i]; 
	}
	return r;
}

int main( int argc, char **argv) {
	int array[TEST_SIZE];
	srand(time(NULL));
	for ( int i = 0; i < TEST_SIZE; ++i) {
		array[i] = rand();
	}

	auto start = std::chrono::high_resolution_clock::now();
	auto res = minmax(array,TEST_SIZE);
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Result (min,max) = " << res.min << ", " << res.max << "\n";
	std::cout << microseconds << "usec \n";

	return 0;
}
