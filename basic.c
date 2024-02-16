#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// input_data, output_data
float train[][2] = {{0, 0}, {1, 5}, {2, 10}, {3, 15}, {4, 20}};

// finding the size of training data number count
const int train_count = sizeof(train) / sizeof(train[0]);

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

float cost(float w, float b) {
  float result = 0.0f;
  for (int i = 0; i < train_count; i++) {
    float x = train[i][0];
    float y = x * w + b;
    float distance = y - train[i][1];
    result += distance * distance;
  }
  result /= train_count;
  return result;
}
int main() {
  srand(time(0));
  float w = rand_float() * 10.0f;
  float b = rand_float() * 5.0f;
  float eps = 1e-3;
  float rate = 1e-3;
  float result = cost(w + eps, b);
  printf("Error result is %f", result);
  for (int i = 0; i < 10000000; i++) {
    float c = cost(w, b);
    float dcost = (cost(w + eps, b) - c) / eps;
    float db = (cost(w, b + eps) - c) / eps;
    b -= rate * db;
    w -= rate * dcost;
  }
  printf("------------------------");
  printf("%f,%f\n", w, w + b);
  return 0;
}
