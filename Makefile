build:
	mpicxx -fopenmp -c parallel.c -o main.o -lm
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o -lm
	nvcc -I./Common  -gencode arch=compute_61,code=sm_61 -c cudaFunctions2.cu -o cudaFunctions2.o -lm
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o cFunctions.o cudaFunctions2.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
	

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP





