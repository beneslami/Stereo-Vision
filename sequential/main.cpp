#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "x86intrin.h"

using namespace cv;
using namespace std;

#define W 3
#define B 45

int main(){
    Mat in_imgL;
    Mat in_imgR;
    unsigned char *inL,*inR, *OutCS;

    int min = 0, temp = 0;
    int step = 255/B;
    int min_position = 0;
    time_t start, end;
    time_t time1, time2;
    int width, height;
    unsigned char outt;

    // LOAD image
    in_imgL = imread("TeddyL.png", 0);
    in_imgR = imread("TeddyR.png", 0);

    inL = in_imgL.data;
    inR = in_imgR.data;

    width = in_imgR.cols;
    height = in_imgR.rows;

    Mat out_img(height,width,CV_8UC1, Scalar(0));
    Mat out_img2(height,width,CV_8UC1, Scalar(0));
    OutCS = &out_img.at<uchar>(0,0);
    start = clock();
    for (int i = 1; i < height; i++)
    {
        for (int j = 1; j < width; j++)
        {
            for(int q = 0; q < B; q++)
            {
                temp = 0;
                temp += abs(*(inR + (i-1)*width + j-1) - *(inL + (i-1)*width + j-1+q));
                temp += abs(*(inR + (i-1)*width + j) - *(inL + (i-1)*width + j+q));
                temp += abs(*(inR + (i-1)*width + j+1) - *(inL + (i-1)*width + j+1+q));
                temp += abs(*(inR + (i)*width + j-1) - *(inL + (i)*width + j-1+q));
                temp += abs(*(inR + (i)*width + j) - *(inL + (i)*width + j+q));
                temp += abs(*(inR + (i)*width + j+1) - *(inL + (i)*width + j+1+q));
                temp += abs(*(inR + (i+1)*width + j-1) - *(inL + (i+1)*width + j-1+q));
                temp += abs(*(inR + (i+1)*width + j) - *(inL + (i+1)*width + j+q));
                temp += abs(*(inR + (i+1)*width + j+1) - *(inL + (i+1)*width + j+1+q));

                if ((temp < min) || (q == 0))
                {
                    min_position = q;
                    min = temp;
                }
            }
            //min_position = min_position/10;
            *(OutCS + (i)*width + j) = (unsigned char) (4 * min_position);
            //out_img.at<uchar>(i,j) = 4*min_position;
        }
    }
    end  = clock();
    time1 = end - start;
    cout << time << endl;
    return 0;    
}
