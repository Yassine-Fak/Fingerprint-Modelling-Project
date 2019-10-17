#include "image.h"
#include <limits>

image::image(unsigned int n=0, unsigned int m=0, unsigned int a=0){
			height = n;
			width = m;
			nb_of_pixel = a;
}

image::image(string image_path){
	matrix = imread(image_path.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
	height = matrix.rows;
	width = matrix.cols;
	nb_of_pixel = height*width;
	path = image_path;
}

image::image(Mat M){
	height = M.rows;
	width = M.cols;
	nb_of_pixel = height*width;
	matrix = M;
}

unsigned int image::width_get(){
	return width;
}

unsigned int image::height_get(){
	return height;
}

Mat image::matrix_get(){
	return matrix;
}

unsigned int image::return_pixel_value(unsigned int x,unsigned int y){
	return matrix.at<uchar>(x,y);
}

void image::save_image(string saving_name){
	imwrite(saving_name.append(".png"),matrix);
	cout << "Image saved." << endl;
}

unsigned int image::return_max_intensity(){
	int max=0;
	for(int i=0;i<this->height_get();i++) {
		for(int j=0;j<this->width_get();j++) {
			if (this->return_pixel_value(i,j)>max) {
				max=this->return_pixel_value(i,j);
			}
		}
	}
	return max;
}

unsigned int image::return_min_intensity(){
	int min=255;
	for(int i=0;i<this->height_get();i++) {
		for(int j=0;j<this->width_get();j++) {
			if (this->return_pixel_value(i,j)<min) {min=this->return_pixel_value(i,j);
			}
		}
	}
	return min;
}

// Creating white squares at a given position
void image::white_square(Coordinates beg,Coordinates end){
	int xb = beg.x_get();
	int yb = beg.y_get();
	int xe = end.x_get();
	int ye = end.y_get();
	// checking if there is an error
	if (xe<xb || ye<yb){
		cout << " Error : The end pixel is smaller than the begin pixel "<< endl;
	}
	else{
		if (xb < width && xe < width && yb < height && ye < height){
			for (int i = xb ; i < xe ; i++){
				for (int j =yb ; j < ye ; j++){
					matrix.at<uchar>(i,j) = 255;
				};
			};
		}
		else {
			cout << " Error : One of the pixels entry values exceed the image size. " << endl;
		};
	};
}

// Creating black squares at a given position
void image::black_square(Coordinates beg,Coordinates end){
	int xb = beg.x_get();
	int yb = beg.y_get();
	int xe = end.x_get();
	int ye = end.y_get();
	// checking if there is an error
	if (xe<xb || ye<yb){
		cout << "Error : The end pixel is smaller than the begin pixel"<<endl;
	}
	else{
		if (xb < height && xe < height && yb < width && ye < width){
			for (int i = xb ; i < xe ; i++){
				for (int j =yb ; j < ye ; j++){
					matrix.at<uchar>(i,j) = 0;
				};
			}
		}
	};
}

// Performing the symmetry along the Y-axis
void image::return_symetry_y(){
	Mat M(this->height_get(),this->width_get(),CV_8UC1,Scalar(0));
	for (int i=0;i<this->height_get();i++){
		for (int j=0;j<this->width_get();j++){
			M.at<uchar>(i,j)=(this->matrix).at<uchar>(this->height_get()-1-i,j);
		}
	}
	matrix = M;
}

// Performing the symmetry along the X-axis
void image::return_symetry_x(){
	Mat M(this->height_get(),this->width_get(),CV_8UC1,Scalar(0));
	for (int i=0;i<this->height_get();i++){
		for (int j=0;j<this->width_get();j++){
			M.at<uchar>(i,j)=(this->matrix).at<uchar>(i,this->width_get()-1-j);
		}
	}
	matrix = M;
}

// The diagonal symmetry which is a combination of the two last symmetries.
void image::return_diagonal_symetry(){
	this->return_symetry_x(); // The first one: Symmetry along the X-axis
	this->return_symetry_y(); // The second one: Symmetry along the Y-axis
}

void image::balance_intensity_exp(Coordinates beg, Coordinates end,unsigned int param_intensity){
	Coordinates center = this->barycenter();
	if (beg.x_get()<=end.x_get() && beg.y_get()<=end.y_get()){
		for (int i=beg.x_get();i<end.x_get()+1;i++){
			for(int j=beg.y_get();j<end.y_get()+1;j++){
				if (matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j) <= param_intensity){
					float r_norm=sqrt(pow(min(beg.x_get() + i - center.x_get(),-beg.x_get() - i + center.x_get()),2)+pow(min(beg.y_get() + j - center.y_get(),-beg.y_get() - j + center.y_get()),2));
					matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j)=
										(int)(exp(-r_norm/75)*matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j));
				}
			}
		}
	}
	else {
		cout << "Error : The end is before the beginning." << endl;
	}
}

void image::balance_intensity_quadratic(Coordinates beg, Coordinates end,unsigned int param_intensity){
	Coordinates center = this->barycenter();
	if (beg.x_get()<=end.x_get() && beg.y_get()<=end.y_get()){
		for (int i=beg.x_get();i<end.x_get()+1;i++){
			for(int j=beg.y_get();j<end.y_get()+1;j++){
				if (matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j) <= param_intensity){
					float r_norm=sqrt(pow(min(beg.x_get() + i - center.x_get(),-beg.x_get() - i + center.x_get()),2)+pow(min(beg.y_get() + j - center.y_get(),-beg.y_get() - j + center.y_get()),2));
					matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j)=
										(int)((25/(25+r_norm))*matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j));
				}
			}
		}
	}
	else {
		cout << "Error : The end is before the beginning." << endl;
	}
}

void image::balance_intensity_normal_2D(Coordinates beg, Coordinates end,unsigned int param_intensity){
	Coordinates barycenter = this->barycenter();
	float x_bary = barycenter.x_get();
	float y_bary = barycenter.y_get();
	vector<float> parameters = this->ellipse_parameters();
	float a = parameters.at(0);
	float b = parameters.at(1);
	if (beg.x_get()<=end.x_get() && beg.y_get()<=end.y_get()){
		for (int i=beg.x_get();i<end.x_get()+1;i++){
			for(int j=beg.y_get();j<end.y_get()+1;j++){
				if (matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j) <= param_intensity){
					//float r_norm=sqrt(pow(min(beg.x_get() + i - center.x_get(),-beg.x_get() - i + center.x_get()),2)+pow(min(beg.y_get() + j - center.y_get(),-beg.y_get() - j + center.y_get()),2));
					float r_norm=sqrt(pow((i-x_bary)/a,2)+pow((j-y_bary)/b,2));
					matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j)=
										ceil(exp(-(pow(r_norm,2)))*matrix.at<uchar>(beg.x_get() + i,beg.y_get() + j));
				}
			}
		}
	}
	else {
		cout << "Error : The end is before the beginning." << endl;
	}
}

Coordinates image::barycenter(){
	unsigned int sum_intensity = 0;
	unsigned int x_sum = 0;
	unsigned int y_sum = 0;
	unsigned int x_bary = 0;
	unsigned int y_bary = 0;
	for (int i = 0 ; i<height ; i++){
		for (int j = 0; j<width ; j++){
			sum_intensity += 255 -  matrix.at<uchar>(i,j);
			x_sum += i*(255 - matrix.at<uchar>(i,j));
			y_sum += j*(255 - matrix.at<uchar>(i,j));
		}
	}
	x_bary = (int)(x_sum/sum_intensity);
	y_bary = (int)(y_sum/sum_intensity);
	Coordinates rep(x_bary,y_bary);
	return rep;
}

void image::print_barycenter(){
	Coordinates p = this->barycenter();
	this->black_square(p -10,p + 10);
}

vector<Coordinates> image::contour(){
	unsigned int threshold = 150;
	Coordinates tmp(0,0);
	vector<Coordinates> rep;
	bool station_gauche;
	bool station_droite;
	for (int i = 1 ; i < height ; i++){
		station_gauche = true;
		station_droite = true;
		for (int j = 0 ; j < width ; j++){
			if (station_gauche && (matrix.at<uchar>(i,j) < threshold)){
				rep.push_back(Coordinates(i,j));
				station_gauche = false;
			}
			if (station_droite && (matrix.at<uchar>(i,width - 1 - j) < threshold)){
				rep.push_back(Coordinates(i,width - 1 - j));
				station_droite = false;
			}
		}
	}
	return rep;
}

void image::print_contour(){
	vector<Coordinates> contour = this->contour();
	float x,y;
	for (int i = 0; i<contour.size(); i++){
		x = (contour.at(i)).x_get();
		y = (contour.at(i)).y_get();
		this->black_square(Coordinates(x,y) - 2,Coordinates(x,y) + 2);
	}
}

vector<float> image::ellipse_parameters(){
	vector<Coordinates> contour = this->contour();
	vector<float> distance_from_bary(contour.size(),0);
	Coordinates barycenter = this->barycenter();
	float norm_min = (contour.at(0) - barycenter).norm();
	float norm_max = (contour.at(0) - barycenter).norm();
	for (int i =1; i < contour.size(); i++){
		distance_from_bary.at(i) = (contour.at(i) - barycenter).norm();
		norm_min = min(distance_from_bary.at(i),norm_min);
		norm_max = max(distance_from_bary.at(i),norm_max);
	}
	vector<float> rep(2,0);
	rep.at(0) = norm_max;
	rep.at(1) = norm_min;
	return rep;
}

vector<float> image::ellipse_parameters_gradient(float epsilon){
	vector<Coordinates> contour = this->contour();
	Coordinates barycenter = this->barycenter();
	vector<float> distance_from_bary(contour.size(),0);
	float norm_min = (contour.at(0) - barycenter).norm();
	float norm_max = (contour.at(0) - barycenter).norm();
	Coordinates var(0,0);
	float i,j;
	float x0 = barycenter.x_get();
	float y0 = barycenter.y_get();
	float dfda,dfdb;
	float errorNorm = 3*epsilon;
	float errorNormPrev = epsilon;
	float a,b;
	int comp = 0;
	float gradientNorm;
	for (int k =1; k < contour.size(); k++){
		distance_from_bary.at(k) = (contour.at(k) - barycenter).norm();
		norm_min = min(distance_from_bary.at(k),norm_min);
		norm_max = max(distance_from_bary.at(k),norm_max);
	}
	a = norm_max;
	b = norm_min;
	while ((comp < 10000)){
		dfda = 0;
		dfdb = 0;
		errorNormPrev = errorNorm;
		errorNorm = 0;
		comp++;
		for (int k = 0; k < contour.size(); k++){
			var  = contour.at(k);
			i = var.x_get();
			j = var.y_get();
			dfda += pow(i - x0,2)*(pow((i - x0)/a,2) + pow((j - y0)/b,2) - 1);
			dfdb += pow(j - y0,2)*(pow((i - x0)/a,2) + pow((j - y0)/b,2) - 1);
			errorNorm += abs(pow((i - x0)/a,2) + pow((i - x0)/a,2) - 1);
		}
		dfda = dfda*(-4/pow(a,3));
		dfdb = dfdb*(-4/pow(b,3));
		gradientNorm = sqrt(pow(dfda,2) + pow(dfdb,2));
		a -= 0.01*dfda/gradientNorm;
		b -= 0.01*dfdb/gradientNorm;
		cout << gradientNorm << endl;
		cout << comp << endl;
	}
	vector<float> rep(2);
	rep.at(0) = a;
	rep.at(1) = b;
	cout << "a = " << a << " and b = " << b << endl;
	return rep;
}

void image::print_ellipse(){
	image printed_image(this->matrix);
	vector<float> parameters = this->ellipse_parameters_gradient(0.10);
	float a = parameters.at(0);
	float b = parameters.at(1);
	Coordinates barycenter = this->barycenter();
	for (int i = 0; i<height; i++){
		for (int j = 0; j<width; j++){
			if (pow((i - barycenter.x_get()),2)/pow(a,2) + pow((j - barycenter.y_get()),2)/pow(b,2) > 0.95
							&& pow((i - barycenter.x_get()),2)/pow(a,2) + pow((j - barycenter.y_get()),2)/pow(b,2) <1.05){
								printed_image.black_square(Coordinates(i,j) - 1,Coordinates(i,j) + 1);
			}
		}
	}
	cout << printed_image;
}

// Transforming an image :
// Preforming a rotation of an image around a given point, and with a given angle
// Here we make a rotation without interpolation
void image::rotation(Coordinates rotation_point, float angle){
	// We create an auxiliary matrix with the same dimensions as the actual one
	Mat mat_aux(height,width, CV_8UC1,Scalar(255));
	float xp,yp;
	// We use two methods of the class Coordiantes
	float a = rotation_point.x_get();
	float b = rotation_point.y_get();
	int x,y;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			xp = (a + (i - a)*cos(-angle) - (j - b)*sin(-angle));
			yp = (b + (i - a)*sin(-angle) + (j - b)*cos(-angle));
			if ((0 <= xp && xp < height) && (0 <= yp && yp < width)){
				x = (int)xp;
				y = (int)yp;
				mat_aux.at<uchar>(i,j) = matrix.at<uchar>(xp,yp);
			}
		}
	}
	matrix = mat_aux;
}

// Now we are going to implement the rotation but using a bilinear interpolation
// In the report we gave the theoretical part of this method
void image::rotation_interpolation_bilinear(Coordinates rotation_point, float angle){
	Mat mat_aux(height,width, CV_8UC1,Scalar(255));
	float xp,yp;
	float a = rotation_point.x_get();
	float b = rotation_point.y_get();
	float delta_fx,delta_fy,delta_fxy;
	int x,y;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			xp = (a + (i - a)*cos(-angle) - (j - b)*sin(-angle));
			yp = (b + (i - a)*sin(-angle) + (j - b)*cos(-angle));
			if ((0 <= xp && xp < height) && (0 <= yp && yp < width)){
				x = (int)xp;
				y = (int)yp;
				delta_fx = float(matrix.at<uchar>(x + 1,y)) - float(matrix.at<uchar>(x,y));
				delta_fy = float(matrix.at<uchar>(x,y + 1)) - float(matrix.at<uchar>(x,y));
				delta_fxy = float(matrix.at<uchar>(x,y)) + float(matrix.at<uchar>(x + 1,y + 1)) - float(matrix.at<uchar>(x + 1,y)) - float(matrix.at<uchar>(x,y + 1));
				mat_aux.at<uchar>(i,j) = delta_fx*(xp - x) + delta_fy*(yp - y) + delta_fxy*(xp - x)*(yp - y) + matrix.at<uchar>(x,y);
			}
		}
	}
	matrix = mat_aux;
}

// To do better, we implemented also the rotation using a bicubic interpolation
// We used an independent function to do the interpolatation
void image::rotation_interpolation_bicubis(Coordinates rotation_point, float angle){
	Mat mat_aux(this->height,this->width, CV_8UC1,Scalar(255));
	float xp,yp;
	float a = rotation_point.x_get();
	float b = rotation_point.y_get();
	int x,y;
	Matrix4d N;
	for (int i = 1; i < this->height-1; i++){
		for (int j = 1; j < this->width-1; j++){
			xp = (a + (i - a)*cos(-angle) - (j - b)*sin(-angle));
			yp = (b + (i - a)*sin(-angle) + (j - b)*cos(-angle));
			if ((1 <= xp && xp < this->height-2) && (1 <= yp && yp < this->width-2)){
				x = (int)xp;
				y = (int)yp;
				for (int m=-1;m<3;m++){
					for (int n=-1;n<3;n++){
						N(m+1,n+1) = matrix.at<uchar>(x+m,y+n);
					}
				}
				// 'interpolation_bicubique' is an independent function defined later on the code
				int res = interpolation_bicubique(N,Coordinates(xp,yp));
				// To normalize we do as following
				if (res >= 255){
					res = 255;
			  }
				if (res <= 0){
					res = 0;
			  }
				mat_aux.at<uchar>(i,j) = res;
			}
		}
	}
	matrix = mat_aux;
}

// In order to interpolate using the bicubic interpolation, we use this independent function
// We gave the theoretical part of this function in the report
int interpolation_bicubique(Matrix4d M,Coordinates P){
	float f00,f01,f02,f03,f10,f11,f12,f13,f20,f21,f22,f23,f30,f31,f32,f33,fy11,fy12,fy21,fy22,fx11,fx12,fx21,fx22,fxy11,fxy12,fxy21,fxy22;
	Matrix4f A;
	Matrix4f B;
	Matrix4f F;
	Matrix4f Coefs;
	A << 1,0,0,0,0,0,1,0,-3,3,-2,-1,2,-2,1,1;
	B = A.transpose();
	float xp,yp;
	xp = P.x_get();
	yp = P.y_get();
	xp -= floor(xp);
	yp -= floor(yp);
	VectorXf X(4);
	VectorXf Y(4);
	f00 = (float)M(0,0);
	f01 = (float)M(0,1);
	f02 = (float)M(0,2);
	f03 = (float)M(0,3);
	f10 = (float)M(1,0);
	f11 = (float)M(1,1);
	f12 = (float)M(1,2);
	f13 = (float)M(1,3);
	f20 = (float)M(2,0);
	f21 = (float)M(2,1);
	f22 = (float)M(2,2);
	f23 = (float)M(2,3);
	f30 = (float)M(3,0);
	f31 = (float)M(3,1);
	f32 = (float)M(3,2);
	f33 = (float)M(3,3);
	fy11 = (f10 - f12)/2;
	fy12 = (f13 - f11)/2;
	fy21 = (f23 - f20)/2;
	fy22 = (f23 - f21)/2;
	fx11 = (f21 - f01)/2;
	fx12 = (f22 - f02)/2;
	fx21 = (f31 - f11)/2;
	fx22 = (f32 - f12)/2;
	fxy11 = fx12 - fx11;
	fxy12 = fy22 - fy12;
	fxy21 = fx22 - fx21;
	fxy22 = (f32 - f22 - fx12)/2;
	F << f11,f12,fy11,fy12,f21,f22,fy21,fy22,fx11,fx12,fxy11,fxy12,fx21,fx22,fxy21,fxy22;
	Coefs = A*F*B;
	X << 1,xp,pow(xp,2),pow(xp,3);
	Y << 1,yp,pow(yp,2),pow(yp,3);
	return int((X.transpose()*Coefs)*Y);
}

// The translation along the X-axis and the Y-axis using the bilinear interpolation
void image::translation(float alpha, float beta){
	Mat mat_aux(height,width,CV_8UC1,Scalar(255));
	float xp,yp;
	float delta_fx,delta_fy,delta_fxy;
	int x,y;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			xp = i - alpha;
			yp = j - beta;
			if ((0 <= xp && xp < height) && (0 <= yp && yp < width)){
				x = (int)xp;
				y = (int)yp;
				delta_fx = matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y);
				delta_fy = matrix.at<uchar>(x,y + 1) - matrix.at<uchar>(x,y);
				delta_fxy = matrix.at<uchar>(x,y) + matrix.at<uchar>(x + 1,y + 1) - matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y + 1);
				mat_aux.at<uchar>(i,j) = delta_fx*(xp - x) + delta_fy*(yp - y) + delta_fxy*(xp - x)*(yp - y) + matrix.at<uchar>(x,y);
			}
		}
	}
	matrix = mat_aux;
}

//in a way to save information, we'll start computing the rotations with bigger images
void image::bigger_image(){
	Mat mat_aux(3*height,3*width, CV_8UC1,Scalar(255));
	for (int i=0; i<height;i++){
		for (int j=0;j<width;j++){
			mat_aux.at<uchar>(i,j)=255;
		}
	}
	for (int i=height; i<2*height;i++){
		for (int j=width;j<2*width;j++){
			mat_aux.at<uchar>(i,j)=matrix.at<uchar>(i-height,j-width);
		}
	}
	for (int i=2*height; i<3*height;i++){
		for (int j=2*width;j<3*width;j++){
			mat_aux.at<uchar>(i,j)=255;
		}
	}
	this->height=3*this->height;
	this->width=3*this->width;
	this->matrix=mat_aux;
}

//to return an image with the exact same dimension of the original one, we have to implement the inverse method of "bigger_image"
void image::smaller_image(){
  Mat mat_aux(int(height/3),int(width/3), CV_8UC1,Scalar(255));
  for (int i=0; i<int(height/3);i++){
		for (int j=0;j<int(width/3);j++){
      mat_aux.at<uchar>(i,j)=matrix.at<uchar>(int(height/3)+i,int(width/3)+j);
    }
  }
  matrix=mat_aux;
	this->height=int(this->height/3);
	this->width=int(this->width/3);
	this->matrix=mat_aux;
}

void image::rotation_warping(Coordinates rotation_center,float radius,float rotation_strength){
	Mat mat_aux(height,width, CV_8UC1,Scalar(255));
	float x0,y0;
	float xp,yp;
	float x,y;
	x0 = rotation_center.x_get();
	y0 = rotation_center.y_get();
	float delta_fx,delta_fy,delta_fxy;
	float dist_from_center;
	float angle;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
				dist_from_center = sqrt(pow(i - x0,2) + pow(j - y0,2));
				if (dist_from_center < radius){
					angle = (-2*PI*rotation_strength/radius)*dist_from_center + 2*PI*rotation_strength;
					xp = (x0 + (i - x0)*cos(-angle) - (j - y0)*sin(-angle));
					yp = (y0 + (i - x0)*sin(-angle) + (j - y0)*cos(-angle));
					if ((0 <= xp && xp < height) && (0 <= yp && yp < width)){
						x = (int)xp;
						y = (int)yp;
						delta_fx = matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y);
						delta_fy = matrix.at<uchar>(x,y + 1) - matrix.at<uchar>(x,y);
						delta_fxy = matrix.at<uchar>(x,y) + matrix.at<uchar>(x + 1,y + 1) - matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y + 1);
						mat_aux.at<uchar>(i,j) = delta_fx*(xp - x) + delta_fy*(yp - y) + delta_fxy*(xp - x)*(yp - y) + matrix.at<uchar>(x,y);
					}
				}
				else{
					mat_aux.at<uchar>(i,j) = matrix.at<uchar>(i,j);
				}
		}
	}
	matrix = mat_aux;
}

void image::translation_warping_x(Coordinates center,Coordinates beg, Coordinates end,float strength_x){
	Mat mat_aux(height,width, CV_8UC1,Scalar(255));
	if ((0 <= center.x_get() && center.x_get() < height) && (0 <= center.y_get() && center.y_get() < width)){
		float dist_from_center;
		int x0 = center.x_get();
		int y0 = center.y_get();
		float delta_fx,delta_fy,delta_fxy;
		float xp,yp;
		float x,y;
		float radius = max(abs(center.x_get() - beg.x_get()),abs(center.x_get() - end.x_get()));
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				dist_from_center = max(abs((i - x0)),abs(j - y0));
				if ((beg.x_get() <= i && i <= end.x_get()) && (beg.y_get() <= j && j <= end.y_get())){
						xp = i - (i - x0)*((-strength_x/radius)*dist_from_center + strength_x);
						yp = j;
						x = (int)xp;
						y = (int)yp;
						if ((0 <= xp && xp < height) && (0 <= yp && yp < width)){
							delta_fx = matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y);
							delta_fy = matrix.at<uchar>(x,y + 1) - matrix.at<uchar>(x,y);
							delta_fxy = matrix.at<uchar>(x,y) + matrix.at<uchar>(x + 1,y + 1) - matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y + 1);
							mat_aux.at<uchar>(i,j) = delta_fx*(xp - x) + delta_fy*(yp - y) + delta_fxy*(xp - x)*(yp - y) + matrix.at<uchar>(x,y);
						}
					}
					else{
						mat_aux.at<uchar>(i,j) = matrix.at<uchar>(i,j);
					}
				}
			}
		}
	else{
		cout << "Pls enter valid coordinates..." << endl;
	}
	matrix = mat_aux;
}

void image::translation_warping_y(Coordinates center,Coordinates beg, Coordinates end,float strength_y){
	Mat mat_aux(height,width, CV_8UC1,Scalar(255));
	if ((0 <= center.x_get() && center.x_get() < height) && (0 <= center.y_get() && center.y_get() < width)){
		float dist_from_center;
		int x0 = center.x_get();
		int y0 = center.y_get();
		float delta_fx,delta_fy,delta_fxy;
		float xp,yp;
		float x,y;
		float radius = max(abs(center.y_get() - beg.y_get()),abs(center.y_get() - end.y_get()));
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				dist_from_center = max(abs((i - x0)),abs(j - y0));
				if ((beg.x_get() <= i && i <= end.x_get()) && (beg.y_get() <= j && j <= end.y_get())){
						xp = i;
						yp = j - (j - y0)*((-strength_y/radius)*dist_from_center + strength_y);
						x = (int)xp;
						y = (int)yp;
						if ((0 <= xp && xp < height) && (0 <= yp && yp < width)){
							delta_fx = matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y);
							delta_fy = matrix.at<uchar>(x,y + 1) - matrix.at<uchar>(x,y);
							delta_fxy = matrix.at<uchar>(x,y) + matrix.at<uchar>(x + 1,y + 1) - matrix.at<uchar>(x + 1,y) - matrix.at<uchar>(x,y + 1);
							mat_aux.at<uchar>(i,j) = delta_fx*(xp - x) + delta_fy*(yp - y) + delta_fxy*(xp - x)*(yp - y) + matrix.at<uchar>(x,y);
						}
					}
					else{
						mat_aux.at<uchar>(i,j) = matrix.at<uchar>(i,j);
					}
				}
			}
		}
	else{
		cout << "Pls enter valid coordinates..." << endl;
	}
	matrix = mat_aux;
}

void image::translation_warping(Coordinates center,Coordinates beg,Coordinates end, float strength_x,float strength_y){
	this->translation_warping_x(center,beg,end,strength_x);
	this->translation_warping_y(center,beg,end,strength_y);
}

// Here the wrap function is a translation along the X-axis
// This method returns the optimal value of the loss function
// it can take time to be executed
int image::detect_trans_along_x_with_first_loss_function(image& image_compare){
	int c = 0;
	float S;
	float min = std::numeric_limits<float>::max();
	Mat matrix_compare = image_compare.matrix_get();
	int p = -height;

	for (int k=0;k<=2*height-1;k++){
		S=0.;
		p+=1;
		if (p>=0) {
			for (int i=0;i<height-p;i++){
				for (int j=0;j<width;j++){
					S+=pow(float(matrix.at<uchar>(i,j))-float(matrix_compare.at<uchar>(i+p,j)),2);
				}
			}
		}
		if (p<0) {
			for (int i=-p;i<height;i++){
				for (int j=0;j<width;j++){
					S+=pow(float(matrix.at<uchar>(i,j))-float(matrix_compare.at<uchar>(i+p,j)),2);
				}
			}
		}
		if (S<min){
			c=p;
			min=S;
		}
	}
	cout << "The minimum value is  : " << min << endl;
	cout << "The estimated value of the translation along the X-axis is : " << c << endl;
	return c;
}

// Here the wrap function is a translation along the X-axis and the Y-axis
// This method returns the optimal value of the loss function which corresponds to the minimum value of p_x and py
// It's a greedy strategy so it may also take time to be executed
void image::detect_trans_along_x_and_y_with_first_loss_function(image& image_compare){
	float S;
	int c_x = 0;
	int c_y = 0;
	int reduc_x=int(height/10);
	int reduc_y=int(width/10);
	int p_x = -height+reduc_x-1;
	int p_y;
	float min=std::numeric_limits<float>::max();
	Mat matrix_compare=image_compare.matrix_get();

	// We start to compute the value of the first loss function
	for (int i=0;i<=2*height-2*reduc_x;i++){
		p_x += 1;
		p_y = -width+reduc_y-1;
		for (int j=0;j<=2*width-2*reduc_y;j++){
			S = 0.;
			p_y += 1;
			if (p_x>=0 && p_y>=0){
				for (int k=0; k<height-p_x;k++){
					for (int l=0;l<width-p_y;l++){
						S += pow((float)(matrix.at<uchar>(k,l))- (float)(matrix_compare.at<uchar>(k+p_x,l+p_y)),2);
					}
				}
			}
			if(p_x>=0 && p_y<0){
				for (int k=0;k<height-p_x;k++){
					for (int l=-p_y;l<width;l++){
						S += pow((float)(matrix.at<uchar>(k,l))-(float)(matrix_compare.at<uchar>(k+p_x,l+p_y)),2);
					}
				}
			}
			if (p_x<0 && p_y>=0){
				for (int k=-p_x;k<height;k++){
					for (int l=0;l<width-p_y;l++){
						S += pow((float)(matrix.at<uchar>(k,l))-(float)(matrix_compare.at<uchar>(k+p_x,l+p_y)),2);
					}
				}
			}
			if (p_x<0 && p_y<0){
				for (int k=-p_x;k<height;k++){
					for (int l=-p_y;l<width;l++){
						S += pow((float)(matrix.at<uchar>(k,l))-(float)(matrix_compare.at<uchar>(k+p_x,l+p_y)),2);
					}
				}
			}
			if (S<min){
				min=S;
				c_x=p_x;
				c_y=p_y;
			}
		}
	}
	cout << "The minimum value is  : " << min << endl;
	cout << "The estimated value of the translation along the X-axis is : " << c_x << endl;
	cout << "The estimated value of the translation along the Y-axis is : " << c_y << endl;
}

// Here the wrap function is a translation along the X-axis and the Y-axis
// This method returns the optimal value of the second loss function which corresponds
void image::detect_trans_along_x_and_y_with_second_loss_function(image& image_compare){
	float f_bar = 0.;
	float gw_bar = 0.;
	float B = 0.;
	float C = 0.;
	float S = 0.;
	int c_x = 0;
	int c_y = 0;
	int reduc_x=int(height/10);
	int reduc_y=int(width/10);
	int p_x = -height+reduc_x-1;
	int p_y;
	float min = std::numeric_limits<float>::min();
	Mat matrix_compare = image_compare.matrix_get();

	// We start by calculating the value of fbar which the average value of the pixel's intensity
	for (int i = 0; i<height; i++){
		for (int j = 0; j< width; j++){
			f_bar += (float)(matrix.at<uchar>(i,j));
		}
	}
	f_bar = f_bar/(float)(nb_of_pixel);

	// Now we compute gwbar witch represents the same thing
	for (int i = 0; i<image_compare.height_get(); i++){
		for (int j = 0; j< image_compare.width_get(); j++){
			gw_bar += (float)(matrix_compare.at<uchar>(i,j));
		}
	}
	gw_bar = gw_bar/(((float)image_compare.height_get())*((float)image_compare.width_get()));

	// B is the sum of the difference between f and fbar to the power two
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			B += pow((float)(matrix.at<uchar>(i,j)) - f_bar,2);
		}
	}
	B = sqrt(B);

	// C is the sum of the difference between g and gwbar to the power two
	for (int i = 0; i < image_compare.height_get(); i++){
		for (int j = 0; j < image_compare.width_get(); j++){
			C += pow((float)(matrix_compare.at<uchar>(i,j)) - gw_bar,2);
		}
	}
	C = sqrt(C);

	// Computation of the value of the second loss function
	for (int i=0;i<=2*height-2*reduc_x;i++){
		p_x += 1;
		p_y = -width+reduc_y-1;
		for (int j=0;j<=2*width-2*reduc_y;j++){
			S = 0.;
			p_y += 1;
			if (p_x>=0 && p_y>=0){
				for (int k=0; k<height-p_x;k++){
					for (int l=0;l<width-p_y;l++){
						S += (((float)(matrix.at<uchar>(k,l)) - f_bar)*((float)(matrix_compare.at<uchar>(k+p_x,l+p_y)) - gw_bar))/(B*C);
					}
				}
			}
			if(p_x>=0 && p_y<0){
				for (int k=0;k<height-p_x;k++){
					for (int l=-p_y;l<width;l++){
						S += (((float)(matrix.at<uchar>(k,l)) - f_bar)*((float)(matrix_compare.at<uchar>(k+p_x,l+p_y)) - gw_bar))/(B*C);
					}
				}
			}
			if (p_x<0 && p_y>=0){
				for (int k=-p_x;k<height;k++){
					for (int l=0;l<width-p_y;l++){
						S += (((float)(matrix.at<uchar>(k,l)) - f_bar)*((float)(matrix_compare.at<uchar>(k+p_x,l+p_y)) - gw_bar))/(B*C);
					}
				}
			}
			if (p_x<0 && p_y<0){
				for (int k=-p_x;k<height;k++){
					for (int l=-p_y;l<width;l++){
						S += (((float)(matrix.at<uchar>(k,l)) - f_bar)*((float)(matrix_compare.at<uchar>(k+p_x,l+p_y)) - gw_bar))/(B*C);
					}
				}
			}
			// Here we maximize this function
			// So we look for the px and py which corresponds to this maximum value
			if (S>min){
				min = S;
				c_x = p_x;
				c_y = p_y;
			}
		}
	}
	cout << "The minimum value is  : " << min << endl;
	cout << "The estimated value of the translation along the X-axis is : " << c_x << endl;
	cout << "The estimated value of the translation along the Y-axis is : " << c_y << endl;
}

// To overload the operator <<, we use the following independent function which we declared as a friend
// As in many other methods we used a method of the library OpenCV
ostream &operator<<(ostream &o, const image & im){
	imshow( "Display window",im.matrix);
	o << "\n" << " Press Enter when you're done! " << "\n";
	waitKey(0);
	return o;
}

float image::detect_rot(image& image_rotation,Coordinates rotation_center,int nb_test){
	float error = this->second_loss_function(image_rotation);
	image depart(this->matrix_get());
	image imtmp = depart;
	float tmp;
	float rot_rep = 0;
	for (int k = 1; k < nb_test; k++){
		imtmp.rotation_interpolation_bicubis(rotation_center,2*PI*k/nb_test);
		tmp = imtmp.second_loss_function(image_rotation);
		if (tmp < error){
			error = tmp;
			rot_rep = 2*PI*k/nb_test;
		}
		imtmp = depart;
		cout << k + 1 << "/" << nb_test << endl;
	}
	depart.rotation_interpolation_bicubis(rotation_center,rot_rep);
	cout << depart;
	cout << rot_rep << endl;
	return rot_rep;
}

vector<float> image::detect_trans(image& image_trans, int nb_testx, int nb_testy){
	float trans_repx;
	float trans_repy;
	float tmpx,tmpy;
	int height = this->height_get();
	int width = this->width_get();
	image depart(this->matrix_get());
	image imtmp = depart;
	float tmp;
	float error = this->second_loss_function(image_trans);
	vector<float> rep(3);
	for (int i = 0; i < nb_testx; i++){
		for (int j = 0; j < nb_testy; j++){
			tmpx = float(height)*float(i)/float(nb_testx);
			tmpy = float(width)*float(j)/float(nb_testy);
			imtmp.translation(tmpx,tmpy);
			tmp = imtmp.second_loss_function(image_trans);
			if (tmp < error){
				error = tmp;
				trans_repx = tmpx;
				trans_repy = tmpy;
			}
			imtmp = depart;
		}
	}
	for (int i = 0; i < nb_testx; i++){
		for (int j = 0; j < nb_testy; j++){
			tmpx = -float(height)*float(i)/float(nb_testx);
			tmpy =  float(width)*float(j)/float(nb_testy);
			imtmp.translation(tmpx,tmpy);
			tmp = imtmp.second_loss_function(image_trans);
			if (tmp < error){
				error = tmp;
				trans_repx = tmpx;
				trans_repy = tmpy;
			}
			imtmp = depart;
		}
	}
	for (int i = 0; i < nb_testx; i++){
		for (int j = 0; j < nb_testy; j++){
			tmpx = float(height)*float(i)/float(nb_testx);
			tmpy = -float(width)*float(j)/float(nb_testy);
			imtmp.translation(tmpx,tmpy);
			tmp = imtmp.second_loss_function(image_trans);
			if (tmp < error){
				error = tmp;
				trans_repx = tmpx;
				trans_repy = tmpy;
			}
			imtmp = depart;
		}
	}
	for (int i = 0; i < nb_testx; i++){
		for (int j = 0; j < nb_testy; j++){
			tmpx = -float(height)*float(i)/float(nb_testx);
			tmpy = -float(width)*float(j)/float(nb_testy);
			imtmp.translation(tmpx,tmpy);
			tmp = imtmp.second_loss_function(image_trans);
			if (tmp < error){
				error = tmp;
				trans_repx = tmpx;
				trans_repy = tmpy;
			}
			imtmp = depart;
		}
	}
	rep.at(0) = trans_repx;
	rep.at(1) = trans_repy;
	rep.at(2) = error;
	return rep;
}

vector<float> image::detect_warp(image& image_compare,int nb_test_rot, int nb_testx, int nb_testy){
	image depart(this->matrix_get());
	image imtmp = depart;
	Coordinates center = depart.barycenter();
	float tmp;
	vector<float> rep(3);
	rep.at(2) = 0;
	vector<float> vtmp(3);
	float rot_rep = 0;
	for (int k = 0; k < nb_test_rot; k++){
		imtmp.rotation_interpolation_bicubis(center, 2*PI*k/nb_test_rot);
		vtmp = imtmp.detect_trans(image_compare,50,50);
		if (vtmp.at(2) < rep.at(2)){
			rep = vtmp;
			rot_rep = 2*PI*k/nb_test_rot;
		}
		imtmp = depart;
		cout << (k + 1)*(100/nb_test_rot) << "%"<< endl;
	}
	rep.at(2) = rot_rep;
	return rep;
}

vector<float> image::detect_warp_small_square(image& image_compare,int nb_test_rot, int square_size){
	image depart(this->matrix_get());
	image imtmp = depart;
	Coordinates center = depart.barycenter();
	float tmp;
	vector<float> rep(3);
	rep.at(2) = 0;
	vector<float> vtmp(3);
	float rot_rep = 0;
	Mat small_square;
	Mat tmp_square;
	Coordinates bary = image_compare.barycenter();
	int halfsize = int(square_size/2);
	image small_image = image_compare.create_small_image(bary,square_size);
	image tmp_image;
	image best_match;
	for (int k = 0; k < nb_test_rot; k++){
		imtmp.rotation_interpolation_bicubis(center, 2*PI*k/nb_test_rot);
		for (int i = halfsize; i < (this->height - halfsize); i++){
			for (int j = halfsize ; j < (this->width - halfsize); j++){
				tmp_image = imtmp.create_small_image(Coordinates(i,j),square_size);
				tmp = small_image.second_loss_function(tmp_image);
				if (tmp < rep.at(2)){
					rep.at(0) = bary.x_get() - float(i);
					rep.at(1) = bary.y_get() - float(j);
					rep.at(2) = tmp;
					rot_rep = 2*PI*k/nb_test_rot;
					best_match = tmp_image;
				}
			}
		}
		cout << (k + 1)*(100/nb_test_rot) << "%"<< endl;
		imtmp = depart;
	}
	rep.at(2) = rot_rep;
	return rep;
}

int image::first_loss_function(image& im2){
	if ((this->height_get() == im2.height_get()) && (this->width_get() == im2.width_get())){
		int height = this->height_get();
		int width = this->width_get();
		Mat mat1 = this->matrix_get();
		Mat mat2 = im2.matrix_get();
		int sum = 0;
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				sum+= pow(mat1.at<uchar>(i,j) - mat2.at<uchar>(i,j),2);
			}
		}
		return sum;
	}
	else {
		cout << "Images arent of the same size." << endl;
		return 100000000;
	}
}

float image::second_loss_function(image& im2){
	int n = im2.nb_of_pixel;
	float moyenne_f = 0;
	float moyenne_g = 0;
	float sum = 0;
	float sum_carre_f = 0;
	float sum_carre_g = 0;
	Mat mat1 = this->matrix_get();
	Mat mat2 = im2.matrix_get();
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			moyenne_f+=float(mat1.at<uchar>(i,j));
			moyenne_g+=float(mat2.at<uchar>(i,j));
		}
	}
	moyenne_f = moyenne_f/n;
	moyenne_g = moyenne_g/n;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			sum+=(float(mat1.at<uchar>(i,j)) - moyenne_f)*(float(mat2.at<uchar>(i,j)) - moyenne_g);
			sum_carre_f+=pow(float(mat1.at<uchar>(i,j)) - moyenne_f,2);
			sum_carre_g+=pow(float(mat2.at<uchar>(i,j)) - moyenne_g,2);
		}
	}
	sum_carre_f = sqrt(sum_carre_f);
	sum_carre_g = sqrt(sum_carre_g);
	return  -sum/((sum_carre_f)*(sum_carre_g));
}

image image::create_small_image(Coordinates point, int image_size){
	Mat matrix = this->matrix_get();
	Mat rep_mat(image_size,image_size, CV_8UC1, Scalar(255));
	int x = point.x_get();
	int y = point.y_get();
	int halfsize = int(image_size/2);
	for (int i = -halfsize; i < halfsize; i++){
		for (int j = -halfsize; j < halfsize; j++){
			rep_mat.at<uchar>(i + halfsize,j + halfsize) = matrix.at<uchar>(x + i,y + j);
		}
	}
	image rep(rep_mat);
	return rep;
}
