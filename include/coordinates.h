#ifndef coordinates_H
#define coordinates_H

/*!
 * \file coordiantes.h
 * \author Nicolas Cavrel - Yassine Fakihani - Ben Sarfati
 * \brief This file contains the C++ functions declaration for the class "coordinates" and all the necessary methods we used to manipulate the position of pixels in the attribute “matrix” of the class “image”.
 */
#include <iostream>
#include <math.h>

/*!
 * \def The macro PI represent the mathematical number pi, we defined it in order to handle rotations of images.
 */
#define PI 3.14159265

/*! \namespace std
 * The C++ Standard Library
 */
using namespace std;

/*! \class Coordinates
* \brief The class "Coordinates" contains one constructor and most of the methods we used to define many functions in the class “image”. It represents the position of one pixel in the attribute matrix accordingly to the basis we choose to do this project.
*/
class Coordinates{

    /*!
    * \brief Declaration of a friend independent function in order to overload the operator “<<” for this class.
    */
    friend ostream &operator<<(ostream &, const Coordinates &);
    private:

          /*!< The attribute x represent the value of the coordinate of the pixel along the X-axis*/
          float x;
          /*!< The attribute y represent the value of the coordinate of the pixel along the Y-axis*/
          float y;

    public:

          /*!
          * \brief This constructor allows us to create an instance of this class by initializing the attributes x and y thanks to the given parameters.
          * \param float_x float_y
          * \return Nothing
          */
          Coordinates(float,float);

          /*!
          * \brief Here we define a getter to know the value of the coordinate along the X-axis.
          * \return The value of the attribute x.
          */
          float x_get();

          /*!
          * \brief Here we define another getter to know the value of the coordinate along the Y-axis.
          * \return The value of the attribute y.
          */
          float y_get();

          /*!
          * \brief We overloaded the operator “+” to be able to add two instances of type “coordinates”.
          * \param Instance_of_type_coordiantes
          * \return The result is an instance of this type, its value is the sum of the two previous ones, coordinate by coordinate.
          */
          Coordinates operator+(Coordinates);

          /*!
          * \brief We overloaded the operator “+” to add the actual instance with an element of type float.
          * \param Float_number
          * \return The result is an instance of this type, its value is the sum of the previous one plus the float number.
          */
          Coordinates operator+(float);

          /*!
          * \brief We overloaded the operator “-” to be able to subtract two instances of type “coordinates”.
          * \param Instance_of_type_coordiantes
          * \return The result is an instance of this type, its value is the subtraction of the two previous ones, coordinate by coordinate.
          */
          Coordinates operator-(Coordinates);
          /*!
          * \brief We overloaded again the operator “-” but this time, we will subtract the actual instance with an element of type float.
          * \param Float_number
          * \return The result is an instance of this type, its value is the subtraction of the previous instance minus the float number.
          */
          Coordinates operator-(float);

          /*!
          * \brief This method calculates the Euclidean norm of an instance of this type, I.e. its distance from the origin of the basis.
          * \param Nothing
          * \return A float which represent the Euclidean norm of an instance.
          */
          float norm();

          /*!
          * \brief This method performs the rotation of a point (I.e. an instance of type Coordinates) around a rotation point and by a given angle, using the formula described in the report.
          * \param Rotation_point Angle
          * \return Nothing, but it modifies the values of the attributes x and y.
          */
          void rotation(Coordinates&,float);

};

#endif
