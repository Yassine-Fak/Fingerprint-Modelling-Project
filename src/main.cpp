#include "image.h"

int main(int argc, char** argv){

 	image im1(argv[1]);
  image im2(argv[1]);
  im1.print_ellipse();
  Coordinates center = im1.barycenter();
  int square_size;
  int i,j,k,l;
  float rot;
  cout << "Rentrez la taille de carre de test : " << endl;
  cin >> square_size;
  cout << "Barycenter : " << im1.barycenter() << endl;
  cout << "Rentrez le point de rotation :" << endl;
  cout << "i : ";
  cin >> i;
  cout << " ; j : ";
  cin >> j;
  cout << endl;
  cout << "Rentrez la valeur de rotation en radian : " << endl;
  cin >> rot;
  im1.rotation_interpolation_bicubis(Coordinates(i,j),rot);
  cout << im1;
  cout << "Rentrez la valeur de translation :" << endl;
  cout << "k : ";
  cin >> k;
  cout << " ; l : ";
  cin >> k;
  cout << endl;
  im1.translation(k,l);
  cout << im1;
  cout << "Pls wait..." << endl;
  vector<float> rep = im2.detect_warp_small_square(im1,10,square_size);
  cout << "Rotation : " << rep.at(2)/PI << " PI" << endl;
  cout << "Translation : " <<  rep.at(0) << " et " << rep.at(1) << endl;
  im2.rotation_interpolation_bicubis(center,rep.at(2));
  im2.translation(rep.at(0),rep.at(1));
  cout << "Best Match" << endl;
  cout << im2;
  cout << "Image qu'il fallait trouver";
  cout << im1;
  return 0;
}
