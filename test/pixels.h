#ifndef pixels_H
#define pixels_H
#define max_intensity 255
#define min_intensity 0
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;


class pixels {
	protected :
		Coordinates p;
		unsigned int intensity_255;
		float intensity_01;
	public :
		pixels(Coordinates,unsigned int);
};

#endif
