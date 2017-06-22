// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image.  In
    particular, this program shows how you can take a list of images from the
    command line and display each on the screen with red boxes overlaid on each
    human face.

    The examples/faces folder contains some jpg images of people.  You can run
    this program on them and see the detections by executing the following command:
        ./face_detection_ex faces/*.jpg

    
    This face detector is made using the now classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  This type of object detector is fairly
    general and capable of detecting many types of semi-rigid objects in
    addition to human faces.  Therefore, if you are interested in making your
    own object detectors then read the fhog_object_detector_ex.cpp example
    program.  It shows how to use the machine learning tools which were used to
    create dlib's face detector. 


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/



#include <iostream>

//using namespace dlib;
using namespace std;

#include "../../include/simple_dlib_face_detection_dll.hpp"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#include "../../include/util/simple_dlib_face_detection_util.hpp"
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../third_party/stb/stb_image.h"
#include "../../third_party/stb/stb_image_write.h"

// ----------------------------------------------------------------------------------------




int main(int argc, char** argv)
{  
	simple_dlib_init("simple_dlib_face_detection.dll");

    try
    {
        if (argc == 1)
        {
            cout << "Give some image files as arguments to this program." << endl;
            return 0;
        }
		void* frontal_face_detector_ptr = get_frontal_face_detector_ptr_dlib();

#ifdef USE_OPENCV
		void* win = create_image_window_dlib();
		resize_image_window_dlib(win, 640, 480);
#endif

        // Loop over all the images provided on the command line.
		for (int i = 1; i < argc; ++i)
		{
			primitive_image_t Img;
			load_primitive_image_dlib(argv[i], Img);

#ifdef USE_OPENCV
			cv::Mat cvImg = pixcels_to_cvmat(Img);
#endif
			std::vector<rectangle_2d_t> rectList;

			face_detector_dlib(Img, (void*)frontal_face_detector_ptr, rectList);
			cout << "Number of faces detected: " << rectList.size() << endl;


#ifdef USE_OPENCV
			std::vector<cv::Mat> facelist;
			cv::Mat cvImg2 = cvImg.clone();
#endif
			for (int j = 0; j < rectList.size(); ++j)
			{
				printf("rect left %d top %d right %d bottom %d\n", (int)rectList[j].left, (int)rectList[j].top, (int)rectList[j].right, (int)rectList[j].bottom);
#ifdef USE_OPENCV
				cv::Point p1((int)rectList[j].left, (int)rectList[j].top);
				cv::Point p2((int)rectList[j].right, (int)rectList[j].bottom);
				try
				{
					cv::rectangle(cvImg, p1, p2, cv::Scalar(0, 0, 255), 1, CV_AA);
					cv::Mat imgSub(cvImg2, cv::Rect(p1, p2));
					facelist.push_back(imgSub);
				}
				catch (...)
				{
					continue;
				}
#endif
			}

#ifdef USE_OPENCV
			//cv::resize(cvImg, cvImg, cv::Size(), 1.0*cvorgImg.cols / cvoutImg.cols, 1.0*cvorgImg.rows / cvoutImg.rows);
			facelist.push_back(cvImg);

			for (int j = 0; j < facelist.size(); ++j)
			{
				char file[256];
				sprintf(file, "zzzz%d_%d.bmp", i, j);
				cv::imwrite(file, facelist[j]);
			}
			primitive_image_t outimg = cvmat_to_pixcels(facelist.back());
			void* dlib_img = primitive_image2dlibImage_dlib(outimg);
			set_image_window_dlib(win, dlib_img);

			delete_dlibImg_dlib(dlib_img);
			delete_primitive_image_dlib(outimg);
#endif

			delete_primitive_image_dlib(Img);
#ifdef USE_OPENCV
			cin.get();
#endif
		}
	}
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

