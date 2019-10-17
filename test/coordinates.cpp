#include "coordinates.h"

Coordinates::Coordinates(float x = 0,float y = 0){
  this->x = x;
  this->y = y;
}

float Coordinates::x_get(){
  return x;
}

float Coordinates::y_get(){
  return y;
}

Coordinates Coordinates::operator+(Coordinates p){
  float a = this->x + p.x;
  float b = this->y + p.y;
  return Coordinates(a,b);
}

Coordinates Coordinates::operator+(float n){
  float a = this->x + n;
  float b = this->y + n;
  return Coordinates(a,b);
}

Coordinates Coordinates::operator-(Coordinates p){
  float a = this->x - p.x;
  float b = this->y - p.y;
  return Coordinates(a,b);
}

Coordinates Coordinates::operator-(float n){
		float a = this->x - n;
		float b = this->y -n;
		return Coordinates(a,b);
}

ostream &operator<<(ostream &o, const Coordinates & p){
	o << "the abscissa value is : " << p.x
	<< " , the ordinate value is : " << p.y << "\n";
	return o;
}

float Coordinates::norm(){
  return sqrt(pow(x,2) + pow(y,2));
}

void Coordinates::rotation(Coordinates& rotation_point,float angle){
  float a = rotation_point.x_get();
  float b = rotation_point.y_get();
  x = a + (x - a)*cos(angle) - (y - b)*sin(angle);
  y = b + (x - a)*sin(angle) + (y - b)*cos(angle);
}

bool Coordinates::is_inside(Coordinates A,Coordinates B,Coordinates C,Coordinates D){
  bool rep = false;
  Coordinates AB = B - A;
  Coordinates AD = D - A;
  Coordinates X = Coordinates(x,y) - A;
  float dotprodXAB = AB.x_get()*X.x_get() + AB.y_get()*X.y_get();
  float dotprodXAD = AD.x_get()*X.x_get() + AD.y_get()*X.y_get();
  if ((0 <= dotprodXAB && dotprodXAB <= 1) && (0 <= dotprodXAD && dotprodXAD <= 1)){
    rep = true;
  }
  return rep;
}
