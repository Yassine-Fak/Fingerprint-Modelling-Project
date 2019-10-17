#ifndef image_H
#define image_H

/*!
* \file image.h
* \author Nicolas Cavrel - Yassine Fakihani - Ben Sarfati
* \brief This file contains the C++ functions declaration for the class "image" we used to do this project, and the declaration for one independent function.
*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "coordinates.h"
#include <algorithm>
#include <Eigen/Dense>

/*! \namespace Eigen
* A C++ template library for linear algebra
*/
using namespace Eigen;
/*! \namespace std
* The C++ Standard Library
*/
using namespace std;
/*! \namespace cv
* The OpenCV (Open Source Computer Vision Library) library
*/
using namespace cv;

/*! \class image
* \brief The class image contains three constructors and most of the functions we defined to respond to many questions.
*/
class image {
		/*!
		* \brief In order to overload the operator “<<” we defined this function, which must be declared as a friend function.
		*/
		friend ostream &operator<<(ostream &, const image &);
		private:
				/*!< The number of pixel in an image*/
				unsigned int nb_of_pixel;
				/*!< The height of an image*/
				unsigned int height;
				/*!< The width of an image*/
				unsigned int width;
				/*!< The matrix of pixels of an image*/
				Mat matrix;
				/*!< The path to the image*/
				string path;

		public:
				/*!
				* \brief Constructor 1: Define an image from its dimensions and number of pixels.
				* \param height width number_of_pixel
				* \return Nothing
				*/
				image(unsigned int, unsigned int, unsigned);

				/*!
				* \brief Constructor 2: Creating an instance of type “image” from its path in the OS.
				* \brief We assume that all of the images we will process are grayscale and we use the OpenCV's methods to initialize all the attributes. For instance, an argument of this class may be: /home/Bureau/UGA/MSIAM-1/Project-Janvier/clean_finger.png
				* \param image_path
				* \return Nothing
				*/
				image(string);

				/*!
				* \brief Constructor 3: We initialize all the attributes from an instance of type Mat.
				* \param Matrix_of_type_Mat
				* \return Nothing
				*/
				image(Mat);

				/*!
				* \brief Here we define a getter to know the width of an image.
				* \return The width of the image
				*/
				unsigned int width_get();

				/*!
				* \brief Here we define another getter to know the height of an image.
				* \return The height of the image
				*/
				unsigned int height_get();

				/*!
				* \brief Another getter to know the values of attribute of type “Mat”.
				* \return The matrix containing the values of the pixels.
				*/
				Mat matrix_get();

				/*!
				* \brief A getter to know the intensity value of the pixel at the coordinate (x, y). If im is an instance of this class. To know the pixel intensity at the position (50,50), we can do: im.return_pixel_value(50,50).
				* \param unsigned_int unsigned_int
				* \return The intensity value Which is of type unsigned int.
				*/
				unsigned int return_pixel_value(unsigned int,unsigned int);

				/*!
				* \brief We create white squares at a given position. Each position is of type “Coordinates”. It raises an error if there no logical values of both parameters. In order to draw a white square in the image im such that the beginning of the square is the point (30,30) and its end is (50,65), we can do: im.white_square(Coordinates(30,30),Coordinates(50,65)).
				* \param Coordinates_beginning Coordinates_end
				* \return It creates a white square on the image.
				*/
				void white_square(Coordinates,Coordinates);

				/*!
				* \brief We create black squares at a given position. Each position is of type “Coordinates”. In order to draw a black square in the image im such that the beginning of the square is the point (30,30) and its end is (50,65), we can do: im.black_square(Coordinates(30,30),Coordinates(50,65)).
				* \param Coordinates_beginning Coordinates_end
				* \return It creates a black square on the image.
				*/
				void black_square(Coordinates,Coordinates);

				/*!
				* \brief This method calculates the maximum intensity value. It uses the method “return_pixel_value” to browse all the elements of attribute “matrix”.
				* \param Nothing
				* \return the maximum intensity of an image
				*/
				unsigned int return_max_intensity();

				/*!
				* \brief This method calculates the minimum intensity of an image.
				* \param Nothing
				* \return the minimum intensity of an image
				*/
				unsigned int return_min_intensity();

				/*!
				* \brief In order to save an image after some modification we define the following method. We save them with the ".png" extension.
				* \param saving_name
				* \return It saves the image in the directory “bin”.
				*/
				void save_image(string);

				/*!
				* \brief Performing the symmetry of an image along the Y-axis.
				* \param Nothing
				* \return It modifies the attributes "matrix" of the original image.
				*/
				void return_symetry_y();

				/*!
				* \brief To perform the symmetry of the image along the X-axis.
				* \param Nothing
				* \return It modifies the attributes "matrix" of the original image.
				*/
				void return_symetry_x();

				/*!
				* \brief The diagonal symmetry which is a combination of the symmetry along the X axis and the one along the Y axis. So this function uses the methods "return_symetry_x" and "return_symetry_y". We can print the transformed image in the screen thanks to the operator “<<”.
				* \param Nothing
				* \return It modifies the attributes "matrix" of the original image.
				*/
				void return_diagonal_symetry();

				/*!
				* \brief This method balances the intensity of the current instance of image throught the exponential function. The user has to provide two coordinates corresponding to the two opposite edge of the rectangle of application. The third parameter is reguling the strentgh of the balancing function : the bigger is param_intensity the stronger is the function.
				* \param beg end param_intensity
				* \return Nothing but modifies the instance on which it is applied.
				*/
				void balance_intensity_exp(Coordinates,Coordinates,unsigned int);

				/*!
				* \brief This method balances the intensity of the current instance of image throught the quadratic function. The user has to provide two coordinates corresponding to the two opposite edge of the rectangle of application. The third parameter is reguling the strentgh of the balancing function : the bigger is param_intensity the stronger is the function.
				* \param beg end param_intensity
				* \return Nothing but modifies the instance on which it is applied.
				*/
				void balance_intensity_quadratic(Coordinates,Coordinates,unsigned int);

				/*!
				* \brief This method balances the intensity of the current instance of image throught the exponential function. This time the method will be applied considering the elliptic shape of the fingerpint. The user has to provide two coordinates corresponding to the two opposite edge of the rectangle of application. The third parameter is reguling the strentgh of the balancing function : the bigger is param_intensity the stronger is the function.
				* \param beg end param_intensity
				* \return Nothing but modifies the instance on which it is applied.
				*/
				void balance_intensity_normal_2D(Coordinates, Coordinates,unsigned int);

				/*!
				* \brief We created this method in order to find the barycenter of an image, because we considered that the barycenter of an image is equals to the center of pressure.
				* \return it returns an instance of type “Coordinates”, its attributes are the location of barycenter in the image.
				*/
				Coordinates barycenter();

				/*!
				* \brief We used the method “barycenter” to know the coordinates of the barycenter, then we applied the function "black_square" to print it.
				* \return It modifies the attribute “matrix” of the image.
				*/
				void print_barycenter();

				/*!
				* \brief This method computes the coordinates of the contour of the finger inside the image. It uses both classes "vector" and "coordinates".
				* \param Nothing
				* \return The result send by this method is a vector containing the coordinates of each pixels representing the contour of the finger.
				*/
				vector<Coordinates> contour();

				/*!
				* \brief This method uses the functions “contour " and “black_square” for the purpose of printing the contour of the finger in the image. In order to print the contour of the finger in an image represented by the instance im, we can do as follow: im.print_contour(). It modifies the attribute matrix of the image. To see the modifications, all what we have to do is to use the overloaded operator “<<”.
				* \param Nothing
				*/
				void print_contour();

				/*!
				* \brief This method estimates the two ellipse parameters a and b using the first naive and unoptimized method (taking the min and max distance from the center of the ellipse).
				* \param None
				* \return Returns a vector of float of size 2, the first element is the estimated value of a and the second the estimated value of b.
				*/
				vector<float> ellipse_parameters();

				/*!
				* \brief This method estimates the two ellipse parameters a and b using the gradient descent method. We need here to provide a step size for the method in the epsilon parameter. The smaller is epsilon and the preciser is the algorithm, but it also makes it slower.
				* \param epsilon
				* \return Returns a vector of float of size 2, the first element is the estimated value of a and the second the estimated value of b.
				*/
				vector<float> ellipse_parameters_gradient(float);

				/*!
				* \brief Computes and prints the best matching ellipse using the gradient descent algorithm. It uses a generic step for the gradient method. This function doesn't modify the current instance of image.
				* \param None
				* \return Nothing but prints the resulting image on screen.
				*/
				void print_ellipse();

				/*!
				* \brief The translation along the X-axis and the Y-axis using the bilinear interpolation. The algorithm we use here to interpolate is like the one in the method “rotation_interpolation_bilinear”. In order to translate an image -represented by the instance im- with 11 pixels following the X-axis and 30 pixels following the Y-axis, we can do as follow: im.translation(11.,30.).
				* \param Float_alpha Float_beta
				* \return It modifies the attribute “matrix”.
				*/
				void translation(float, float);

				/*!
				* \brief Thanks to this function we can effectuate a rotation of the image around a point and with a specific angle, both given as parameters. Here we interpolate using the bilinear interpolation (In the report we detailed the theoretical calculation) which is quite good, but not as good as the interpolation with the bicubic method. To rotate an image -represented by the instance im- around its barycenter and with an angle PI/4, we can do as follow: im.rotation_interpolation_bilinear(im.barycenter(),PI/4)
				* \param Rotation_point Angle
				* \return It modifies the attribute “matrix”, we can use the operator “<<” to show the modified image.
				*/
				void rotation_interpolation_bilinear(Coordinates,float);

				/*!
				* \brief To do a better rotation of an image around a point and with a specific angle, we implemented this method where we interpolate using the bicubic interpolation thanks to the independent function “interpolation_bicubique”. Also, inside this method we normalize all the values resulting from the execution of the independent function. To rotate an image -represented by the instance im- around its barycenter and with an angle PI/4, we can do as follow: im.rotation_interpolation_bicubis(im.barycenter(),PI/4)
				* \param Rotation_point Angle
				* \return Nothing, but it modifies the attributes “matrix”.
				*/
				void rotation_interpolation_bicubis(Coordinates, float);

				/*!
				* \brief Performs the rotation warping of an image, which is a rotation decreasing of strength the further away you are from the center. This center is defined by the rotation_center parameter. A negative rotation_strength is perfoming the rotation in the other way. This method uses a bilinear interpolation to do so.
				* \param rotation_center radius rotation_strength
				* \return Nothing, but it modifies the attributes “matrix”.
				*/
				void rotation_warping(Coordinates,float,float);

				/*!
				* \brief Performs the translation warping according the Ox axis, which is a stretching of the image. You can choose either to compress or to zoom on the image by changing the positivity of the strength of the strength_x parameter. The parameter center defines the center of the stretching and the parameters beg and end the zone of application of the function.
				* \param center beg end strength_x
				* \return Nothing, but it modifies the attributes “matrix”.
				*/
				void translation_warping_x(Coordinates,Coordinates,Coordinates,float);

				/*!
				* \brief Similar to the translation_warping_x funtion but applied to the Oy axis.
				* \param center beg end strength_y
				* \return Nothing, but it modifies the attributes “matrix”.
				*/
				void translation_warping_y(Coordinates,Coordinates,Coordinates,float);

				/*!
				* \brief This function allows to apply the functions translation_warping_x and translation_warping_y one after this other.
				* \param center beg end strength_x strength_y
				* \return Nothing, but it modifies the attributes “matrix”.
				*/
				void translation_warping(Coordinates,Coordinates,Coordinates,float,float);

				/*!
				* \brief In a way to save information, and ovoid any loss of pixels intensity when making a rotation, we'll start computing rotations with bigger images, then we call the method “smaller_image” to come back to the normal dimensions of the image. If im is an instance of this class, we can do like this to make bigger this image: im.bigger_image()
				* \param Nothing
				*/
				void bigger_image();

				/*!
				* \brief After calling the method "bigger_image" we need to return an image with the exact same dimension of the original one. That’s why we must implement the inverse method which is “smaller_image”.
				* \param Nothing
				* \return Nothing, but it modifies the attributes “matrix”.
				*/
				void smaller_image();

				/*!
				*  \brief This function deal with the case where the wrap function is a translation along the X-axis such that there is only a single translation parameter px to estimate. Here we use the first lost function. We didn’t call the one defined independently named “first_loss_function”. it can take time to be executed. If im1 and im2 are two instances representing the same image, in order to test this function, we can do like this: im2.translation(4.,0.) then im1.detect_trans_along_x_with_first_loss_function(im2) and the result returned by this method will be 4! Here you can test also negative values!
				*  \param image
				*  \return An integer which represent the estimated parameter px.
				*/
				int detect_trans_along_x_with_first_loss_function(image&);

				/*!
				*  \brief This function deal with the case where the wrap function is a translation along the X-axis and Y-axis such that there are two translation parameters px and py to estimate. Here we use the first loss function. We didn’t call the one defined independently named “first_loss_function”. The execution of this function may take a lot of time because it’s a greedy startegy. If im1 and im2 are two instances representing the same image. In order to test this function, we can do like this: im2.translation(10.,11.) then  im1.detect_trans_along_x_and_y_with_first_loss_function(im2) and the result printed on the screen will be (10.,11.) ! Here you can test also negative values!
				*  \param image
				*  \return Nothing but it prints on the screen the estimated parameter px and py.
				*/
				void detect_trans_along_x_and_y_with_first_loss_function(image&);

				/*!
				*  \brief The execution of this function may take a lot of time because it’s a greedy strategy. This function is like the one called “detect_trans_along_x_and_y_with_first_loss_function”, because it deals with the case where the wrap function is a translation along the X-axis and Y-axis such that there are two translation parameters px and py to estimate. Its advantage is that it uses (implicitly) the second loss function. So, we didn’t call the method named “second_loss_function”. Finally, it's worth noting that in this method we maximize the second loss function and we do not minimize it like we did in the method “detect_trans_along_x_and_y_with_first_loss_function”. If im1 and im2 are two instances representing the same image, in order to test this function, we can do like this: im2.translation(10.,11.) then  im1.detect_trans_along_x_and_y_with_second_loss_function(im2) and the result printed on the screen will be (10.,11.) ! Here you can test also negative values!
				*  \param image
				*  \return Nothing but it prints on the screen the estimated parameter px and py.
				*/
				void detect_trans_along_x_and_y_with_second_loss_function(image&);

				/*!
				* \brief Here we do a transformation of an image using the rotation around a point and with an angle without any interpolation, using the mathematical formula we gave in the report and two methods of the class coordiantes. To rotate an image -represented by the instance im- around its barycenter and with an angle PI/4, we can do as follow: im.rotation(im1.barycenter(),PI/4).
				* \param Rotation_point Angle
				* \return Nothing, but it modifies the attributes “matrix”.
				*/
				void rotation(Coordinates,float);

				/*!
				*  \brief This function detects if the current instance of image is the rotation of the image contained in the image_rotation parameter. We also have to specify where we want the rotation to happen with the rotation_center parameter. As it is impossible in general to find the exact rotation of a picture and another (if the image is a rotation of exactly sqrt(2) then it would take an infinite time to compute the exact rotation value) we have to specify the number of step we want to do with the pameter nb_test : the more step and the more accurate is the result, but the slower is the algorithm...
				*  \param image_rotation rotation_center nb_test
				*  \return Returns a float representing the rotation angle in radius.
				*/
				float detect_rot(image&,Coordinates,int);

				/*!
				*  \brief This function detect if the current instance of image is a translation of the one contained in the image_trans parameter. As for the detect_rot function, we have to define the number of operations we want to perform along the Ox and Oy axis. This can be done by inputing it in the parameters nb_testx and nb_testy.
				*  \param image_trans nb_testx nb_testy
				*  \return Returns a vector of 3 floats, the two first are respectively the translation answer along the Ox and Oy axis and the last one the error value.
				*/
				vector<float> detect_trans(image&,int,int);

				/*!
				*  \brief We define here the first loss function which is the sum of squared errors between the pixels of two images. The sum is taken over all pixels of images.
				*  \param image
				*  \return An integer which represent the sum taken over all pixels of both images.
				*/
				int first_loss_function(image&);

				/*!
				*  \brief We define here the second loss function where we calculate the mean of the pixel's intensity of both images. We browse all the pixels of both images.
				*  \param image
				*  \return An integer which represent the sum taken over all pixels of both images.
				*/
				float second_loss_function(image&);

				/*!
				*  \brief This function detects if the current instance of image is a warp of the image contained in the parameter image_compare using the first naive method (trying out every rotation/translation and computing the error on the whole image). We have to profide the number of test we want to perform on the rotation, on the transalations along the Ox and Oy axises respectively in the nb_test_rot, nb_testx and nb_testy parameters. Once again, the bigger are these numbers and the slower is the algorithm but the more accurate is the result.
				*  \param image_compare nb_test_rot nb_testx nb_testy
				*  \return Returns a vector of float of size 3, the two first parameter are the two translation answers (along Ox and Oy) and the last is the rotation answer.
				*/
				vector<float> detect_warp(image&,int,int,int);

				/*!
				*  \brief This function detects if the current instance of image is a warp of the image contained in the parameter image_compare using optimized small square method (we here try to match only a small square of the image). In this method, we just have to give the number of test we want to perform on the rotation throught the nb_test_rot parameter. We no longer need the number of test for the translation since the efficiency of the algorithm allows us to test every integer values for the translation. We can also set the size of the test square with square_size parameter (in number of pixels).
				*  \param image_compare nb_test_rot square_size
				*  \return Returns a vector of float of size 3, the two first parameter are the two translation answers (along Ox and Oy) and the last is the rotation answer.
				*/
				vector<float> detect_warp_small_square(image&,int,int);

				/*!
				*  \brief Auxiliary function for the detect_warp_small_square method. It helps creating the small piece of image to compare with the test square. It will create a square image centered on the coordinate point an of size image_size.
				*  \param point image_size
				*  \return Returns an image as described above.
				*/
				image create_small_image(Coordinates,int);

};

/*!
*  \brief In order to interpolate using the bicubic interpolation, we use this independent function. We use here some classes of the library “Eigen”. We detailed all the theoretical calculation in the report.
*  \param Matrix4d_M Coordinates_P
*  \return The interpolated point.
*/
int interpolation_bicubique(Matrix4d,Coordinates);

#endif
