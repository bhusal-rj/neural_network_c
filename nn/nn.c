#include "nn.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

Mat mat_alloc(int rows, int cols) {
  // this function attocates the memory space for the matrix with the help of
  // malloc standard function
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.es = (float *)malloc(sizeof(*m.es) * rows * cols);
  return m;
}

void mat_dot(Mat dst, Mat a, Mat b) {
  // helps in dot product of the matrix
  assert(a.cols == b.rows);
  size_t n = a.cols;
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
}

void mat_rand(Mat m) {
  // create the random matrix with random data
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = rand_float();
    }
  }
}

void mat_fill(Mat m, float x) {
  // fill the random given data to the matrix
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = x;
    }
  }
}

void mat_sum(Mat dst, Mat a) {
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_print(Mat m, char *name) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
}

Mat mat_row(Mat m, int row) {
  // create the single row from the given matrix
  return (Mat){
      .rows = 1,
      .cols = m.cols,
      .stride = m.stride,
      .es = &MAT_AT(m, row, 0),
  };
}

void mat_copy(Mat dst, Mat src) {
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

void mat_sig(Mat m) {
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
    }
  }
}

void neural_print(NN nn, const char *name) {
  printf("Printing the neural network\n");
  for (int i = 0; i < nn.count; i++) {
    printf("ws[%d]=[ ", i);
    MAT_PRINT(nn.ws[i]);
    printf("]\n");
    printf("bs[%d]=[ ", i);
    MAT_PRINT(nn.bs[i]);
    printf("]\n");
  }
}

void nn_rand(NN nn) {

  for (int i = 0; i < nn.count; i++) {
    mat_rand(nn.bs[i]);
    mat_rand(nn.ws[i]);
  }
}

void nn_forward(NN nn) {
  for (int i = 0; i < nn.count; i++) {
    mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
    mat_sum(nn.as[i + 1], nn.bs[i]);
    mat_sig(nn.as[i + 1]);
  }
}
NN nn_alloc(int *arch, int count) {
  assert(count > 0);
  NN nn;
  nn.count = count - 1;
  nn.ws = malloc(sizeof(*nn.ws) * (nn.count));
  nn.bs = malloc(sizeof(*nn.bs) * (nn.count));
  nn.as = malloc(sizeof(*nn.as) * (nn.count + 1));

  nn.as[0] = mat_alloc(1, arch[0]);
  for (int i = 1; i < count; i++) {
    nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
    nn.bs[i - 1] = mat_alloc(1, arch[i]);
    nn.as[i] = mat_alloc(1, arch[i]);
  }
  return nn;
}


float nn_cost(NN nn, Mat ti, Mat to) {
  assert(ti.rows == to.rows);
  assert(to.cols == NN_OUTPUT(nn).cols);
  int n = ti.rows;

  float cost = 0;
  for (int i = 0; i < n; i++) {
    Mat x = mat_row(ti, i);
    Mat y = mat_row(to, i);
    mat_copy(NN_INPUT(nn), x);
    nn_forward(nn);
    int q = to.cols;
    for (int j = 0; j < q; j++) {
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
      cost += d * d;
    }
  }
  return cost / n;
}
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to) {
  float saved;
  float c = nn_cost(nn,ti,to);
  for(int i=0;i<nn.count;i++){
    for(int j=0;j<nn.ws[i].rows;j++){
      for(int k=0;k<nn.ws[i].cols; k++){
        saved = MAT_AT(nn.ws[i],j,k);
        MAT_AT(nn.ws[i],j,k) += eps;
        MAT_AT(g.ws[i],j,k)=(nn_cost(nn,ti,to) - c)/eps;
        MAT_AT(nn.ws[i],j,k)= saved;
      }
    }
    for(int j=0;j<nn.bs[i].rows;j++){
      for(int k=0;k<nn.bs[i].cols; k++){
        saved = MAT_AT(nn.bs[i],j,k);
        MAT_AT(nn.bs[i],j,k) += eps;
        MAT_AT(g.bs[i],j,k)=(nn_cost(nn,ti,to) - c)/eps;
        MAT_AT(nn.bs[i],j,k)= saved;
      }
    }
  }
}
void nn_learn(NN nn, NN g , float rate){
  for(int i=0;i<nn.count;i++){
    for(int j=0;j<nn.ws[i].rows;j++){
      for(int k=0;k<nn.ws[i].cols;k++){
        MAT_AT(nn.ws[i],j,k)-= rate * MAT_AT(g.ws[i],j,k);
      }
    }

    for(int j=0;j<nn.bs[i].rows;j++){
      for(int k=0;k<nn.bs[i].cols;k++){
        MAT_AT(nn.bs[i],j,k)-= rate * MAT_AT(g.bs[i],j,k);
      }
    }
  }
}
