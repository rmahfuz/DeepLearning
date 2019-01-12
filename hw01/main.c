#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int calc(int*, int*);
int c_conv(int, int, int, int);

int main() {
	int start, stop, end, i, oprs = 0;
	for(i = 0; i <= 11; i++){
		start = clock();
		oprs = c_conv(3, pow(2.0, (double)i), 3, 1);
		stop = clock();
		printf("i = %d, time taken: %d\n", i, stop-start);
	}
	return 0;
}

int c_conv(int in_channel, int o_channel, int kernel_size, int stride) {
  int cnt, i, oprs = 0;
  int* image = malloc(sizeof(int)*60*110);
  int* kernel = malloc(sizeof(int)*3*3);
  for(cnt = 0; cnt < o_channel; cnt++) {
    //generate the kernel:
    for(i = 0; i < kernel_size*kernel_size; i++) {
	    *(kernel + i*sizeof(int)) = rand();
    }
    //red channel:
    for(i = 0; i < 60*110; i++) {
	    *(image + i*sizeof(int)) = rand();
    }
    oprs += calc(image, kernel);
    //green channel:
    for(i = 0; i < 60*110; i++) {
	    *(image + i*sizeof(int)) = rand();
    }
    oprs += calc(image, kernel);
    //blue channel:
    for(i = 0; i < 60*110; i++) {
	    *(image + i*sizeof(int)) = rand();
    }
    oprs += calc(image, kernel);;
    }
    free(image); free(kernel);
    return oprs;
}

int calc(int* img, int* kernel)
{
	int i, j, k, l, ans, oprs = 0;
	for(i = 0; i < (60-3); i++) { //go down
		//row = []
		for(j = 0; j < (110-3); j++) { //go right
			ans = 0;
			for(k = 0; k < 3; k++) {
				//row = []
				for(l = 0; l < 3; l++) {
					//ans += kernel[k][l] * img[i+k][j+l];
					ans += (*(kernel + (i*3) + j) * *(img + ((i+k)*110) + (j+l)));
					oprs += 17; //9 multiplication + 8 addition
				}
			}
		}
	}
	return oprs;
}
/*
For a 50x100 pixel image,
i = 0, time taken: 0
i = 1, time taken: 0
i = 2, time taken: 0
i = 3, time taken: 10000
i = 4, time taken: 10000
i = 5, time taken: 20000
i = 6, time taken: 50000
i = 7, time taken: 100000
i = 8, time taken: 200000
i = 9, time taken: 400000
i = 10, time taken: 790000
i = 11, time taken: 1580000

For a 60x110 pixel image,
i = 0, time taken: 0
i = 1, time taken: 0
i = 2, time taken: 0
i = 3, time taken: 10000
i = 4, time taken: 20000
i = 5, time taken: 30000
i = 6, time taken: 70000
i = 7, time taken: 130000
i = 8, time taken: 260000
i = 9, time taken: 520000
i = 10, time taken: 1050000
i = 11, time taken: 2110000

*/ 
