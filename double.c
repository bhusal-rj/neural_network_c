#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float sigf(float x){
  return 1.f / (1.f + exp(-x));
}
// OR Gate
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

const int train_count = sizeof(train) / sizeof(train[0]);

float cost(float w1, float w2, float b) {
  float result = 0.0f;
  for (int i = 0; i < train_count; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = sigf( x1 * w1 + x2 * w2 + b);
    float d = y - train[i][2];
    result += d * d;
  }
  result /= train_count;
  return result;
}

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }
int main() {
  // for seeding the random number
  srand(time(NULL));
  int epoch = 10000;
  float w1 = rand_float();
  float w2 = rand_float();
  float b = rand_float();
  float eps = 1e-3;
  float rate = 1e-3;
  printf("Weight 1 :- %f and Weight 2 :- %f", w1, w2);
  float c = cost(w1, w2, 1);
  printf("The cost function is %f", c);
  for (int i = 0; i < epoch; i++) {

    float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
    float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
    w1 -= rate * dw1;
    w2 -= rate * dw2;
  }
  printf("\nThe weights are %f and %f",w1,w2);
}
