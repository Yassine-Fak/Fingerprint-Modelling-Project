#ifndef coordinates_H
#define coordinates_H
#include <iostream>
#include <math.h>

#define PI 3.14159265

using namespace std;


class Coordinates{

// Declaration of  a friend independent function in order to overload the operator <<
friend ostream &operator<<(ostream &, const Coordinates &);
private:

  float x;
  float y;

public:
  Coordinates(float,float);
  float x_get();
  float y_get();
  Coordinates operator+(Coordinates);
  Coordinates operator+(float);
  Coordinates operator-(Coordinates);
  Coordinates operator-(float);
  float norm();
  void rotation(Coordinates&,float);
  bool is_inside(Coordinates,Coordinates,Coordinates,Coordinates);

};

#endif
