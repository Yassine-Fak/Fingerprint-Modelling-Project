#ifndef image_H
#define image_H

/*!
 * \file image.h
 * \brief Lecteur de musique de base
 * \author nicolas Cavrel - Yassine Fakihani - Ben Sarfati
 */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "coordinates.h"
#include <algorithm>
#include <Eigen/Dense>

/*! \namespace Eigen
*
* A C++ template library for linear algebra
*
*/
/*! \namespace cv
*
* The OpenCV (Open Source Computer Vision Library) library
*
*/
/*! \namespace std
 *
 * The C++ Standard Library
 *
 */
using namespace Eigen;
using namespace std;
using namespace cv;

/*! \class image
*
* \brief class representing an image
*
*/
class image {
friend ostream &operator<<(ostream &, const image &);
private:

		unsigned int nb_of_pixel; /*!< The number of pixel in an image*/
		unsigned int height; /*!< The height of an image*/
		unsigned int width; /*!< The width of an image*/
		Mat matrix; /*!< The matrix of pixels of an image*/
		string path; /*!< The path of pixels of an image*/

public:
	 /*!
	 *  \brief Constructeur1
	 *
	 *  definir une image a partir de ses dim
	 *
	 *  \param height, widh, qch
	 */
		image(unsigned int, unsigned int, unsigned);
		/*!
		*  \brief Constructeur2
		*
		*  definir une image a partir de son chemin
		*
		*  \param path
		*/
		image(string);
		image(Mat);
		unsigned int height_get();
		unsigned int width_get();
		Mat matrix_get();
		unsigned int return_pixel_value(unsigned int,unsigned int);
		void white_square(Coordinates,Coordinates);

		void black_square(Coordinates,Coordinates);
		/*!
		 *  \brief renvoie la valeur maximal d'intensity
		 *
		 *  Methode qui permet de trouver la valeur maximal d'intensite
		 *
		 *  \return la valeur maximal d'intensite
		 */
		unsigned int return_max_intensity();
		unsigned int return_min_intensity();
		/*!
		 *  \brief cela permet de sauvegarder une image apres l'avoir modifier
		 *
		 */
		void save_image(string);
		void return_symetry_y();
		void return_symetry_x();
		void return_diagonal_symetry();
		void print_pressure_center(unsigned int);
		Coordinates return_pressure_center(unsigned int);
    void balance_intensity_exp(Coordinates,Coordinates,unsigned int);
   	void balance_intensity_quadratic(Coordinates,Coordinates,unsigned int);
		Coordinates barycenter();
		void print_barycenter();
		vector<Coordinates> contour();
		void print_contour();
		vector<float> ellipse_parameters();
		vector<float> ellipse_parameters_gradient(float);
		Coordinates ellipse_max_dist_from_bary();
		void print_ellipse();
		void balance_intensity_normal_2D(Coordinates, Coordinates,unsigned int);
		void translation(float, float);
		void rotation_interpolation_bilinear(Coordinates,float);
		void rotation_interpolation_bicubis(Coordinates, float);
		void rotation_warping(Coordinates,float,float);
		void translation_warping_x(Coordinates,Coordinates,Coordinates,float);
		void translation_warping_y(Coordinates,Coordinates,Coordinates,float);
		void translation_warping(Coordinates,Coordinates,Coordinates,float,float);
		void bigger_image();
		void smaller_image();
		int argmin_error_x(image);
		int argmin_error_xy(image);
		int argmin_error_xy_bis(image);
		int argmin_error_xy_rot(image);
		void rotation(Coordinates,float);
		float detect_rot(image,Coordinates,int);
		vector<float> detect_trans(image,int,int);
		int first_loss_function(image);
		float second_loss_function(image);
		void detect_warp(image,Coordinates,int,int,int);

};

/*!
 *  \brief renvoie la valeur de l'interpolation bicibic
 *
 *  Methode qui permet de trouver la valeur maximal d'intensite
 *  \param Matric4*4 point
 *
 *  \return la valeur maximal d'intensite
 */
int interpolation_bicubique(Matrix4d,Coordinates);

#endif
