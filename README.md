#compile reduce on Stampede:
  nvcc -arch=compute_35 -code=sm_35 -o reduce.out reduce.cu <br />
  "inp.txt" is test input file.
