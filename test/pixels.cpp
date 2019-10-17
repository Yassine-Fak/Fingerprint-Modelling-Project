#include "pixels.h"



pixels::pixels(Coordinates p1, unsigned int n){																// constructor 
			p = p1;
			intensity_255 = n;
			intensity_01 = n/max_intensity;
}

