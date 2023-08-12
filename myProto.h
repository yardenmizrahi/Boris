#pragma once

#define COLS 5
#define MASTER 0
#define MAX_LEN 100 // Max length of each string

double computeX(double x1, double x2, double t);
double computeY(double x, double a, double b);
double euclideanDistance(double x1, double y1, double x2, double y2);
int satisfiesProximityCriteria(int currentIndex, double t, double D, int N, double* x1, double* x2, double* a, double* b , int K);
int computeOnGPU(char h_strings[MAX_LEN][MAX_LEN], int currentIndex, double D, int N, int K, double TCount, double* x1, double* x2, double* a, double* b, int* id);
