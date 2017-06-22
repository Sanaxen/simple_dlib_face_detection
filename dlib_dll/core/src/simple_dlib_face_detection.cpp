#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>


#include <iostream>
#include <omp.h>

using namespace dlib;
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../third_party/stb/stb_image.h"
#include "../../../third_party/stb/stb_image_write.h"

#include "../../core/include/simple_dlib_face_detection.hpp"

primitive_image_t primitive_image_clone(primitive_image_t& p)
{
	primitive_image_t q = p;
	q.pixcels = new unsigned char[p.x*p.y*p.channels];

	const int sz = p.x*p.y*p.channels;
#pragma omp parallel for
	for (int i = 0; i < sz; i++) q.pixcels[i] = p.pixcels[i];

	return q;
}

void delete_primitive_image(primitive_image_t& p)
{
	if (p.pixcels) delete[] p.pixcels;
}

void delete_dlibImg(void* dlibImg)
{
	matrix<bgr_pixel>* dlibimage = (matrix<bgr_pixel>*)dlibImg;
	delete dlibimage;
}


int ToBMP(const char* filename, char* newfile)
{
	unsigned char *image_data = 0;
	int x, y;
	int nbit;
	image_data = stbi_load(filename, &x, &y, &nbit, 0);
	if (image_data == NULL)
	{
		printf("image file[%s] read error.\n", filename);
		return -1;
	}
	stbi_write_bmp(newfile, x, y, nbit, (void*)image_data);
	return 0;
}

void* get_frontal_face_detector_ptr()
{
	frontal_face_detector* detector = new frontal_face_detector(get_frontal_face_detector());

	return (void*)detector;
}


void load_primitive_image(char* filename, primitive_image_t& p)
{
	p.pixcels = stbi_load(filename, &p.x, &p.y, &p.channels, 0);
	if (p.pixcels == NULL)
	{
		printf("image file[%s] read error.\n", filename);
		return;
	}
}

void save_primitive_image(char* filename, primitive_image_t p)
{
	stbi_write_bmp(filename, p.x, p.y, p.channels, (void*)p.pixcels);
}

primitive_image_t dlibImage2primitive_image(void* img_p, const int channels)
{
	matrix<bgr_pixel> img = *((matrix<bgr_pixel>*)img_p);

	primitive_image_t out_image;
	out_image.x = img.nc();
	out_image.y = img.nr();
	out_image.channels = channels;
	out_image.pixcels = new unsigned char[img.nr()*img.nc()*channels];

	const int sz = img.nr()*img.nc();
#pragma omp parallel for
	for (int i = 0; i < sz; i++)
	{
		if (channels == 1)
		{
			out_image.pixcels[channels*i + 0] = img(i).red;
		}
		else
		{
			out_image.pixcels[channels*i + 0] = img(i).red;
			out_image.pixcels[channels*i + 1] = img(i).green;
			out_image.pixcels[channels*i + 2] = img(i).blue;
		}
	}
	return out_image;
}

void* primitive_image2dlibImage(primitive_image_t& image)
{
	matrix<bgr_pixel>* img_p = new matrix<bgr_pixel>(image.y, image.x);

	const int channels = image.channels;

	const int sz = image.x*image.y;
	if (channels == 1)
	{
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			bgr_pixel bgr(image.pixcels[channels*i + 0], image.pixcels[channels*i + 0], image.pixcels[channels*i + 0]);
			(*img_p)(i) = bgr;
		}
		return (void*)img_p;
	}

#pragma omp parallel for
	for (int i = 0; i < sz; i++)
	{
		bgr_pixel bgr(image.pixcels[channels*i + 2], image.pixcels[channels*i + 1], image.pixcels[channels*i + 0]);
		(*img_p)(i) = bgr;
	}
	return (void*)img_p;
}

void primitive_image2dlibImage_(primitive_image_t& image, matrix<bgr_pixel>& img_p)
{

	const int channels = image.channels;

	const int sz = image.x*image.y;
	if (channels == 1)
	{
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			bgr_pixel bgr(image.pixcels[channels*i + 0], image.pixcels[channels*i + 0], image.pixcels[channels*i + 0]);
			img_p(i) = bgr;
		}
		return;
	}
#pragma omp parallel for
	for (int i = 0; i < sz; i++)
	{
		bgr_pixel bgr(image.pixcels[channels*i + 2], image.pixcels[channels*i + 1], image.pixcels[channels*i + 0]);
		img_p(i) = bgr;
	}

}

int face_detector(primitive_image_t& image, void* detector_ptr, std::vector<rectangle_2d_t>& rectangleList)
{
	frontal_face_detector& detector = *((frontal_face_detector*)detector_ptr);

	matrix<bgr_pixel> img_p(image.y, image.x);
	primitive_image2dlibImage_(image, img_p);

	//save_bmp(img_p, "zzzz.bmp"); exit(0);
	//primitive_image_t q = dlibImage2primitive_image((void*)&img_p, 3);
	//save_primitive_image("zzzz.bmp", *q);


	// Make the image bigger by a factor of two.  This is useful since
	// the face detector looks for faces that are about 80 by 80 pixels
	// or larger.  Therefore, if you want to find faces that are smaller
	// than that then you need to upsample the image as we do here by
	// calling pyramid_up().  So this will allow it to detect faces that
	// are at least 40 by 40 pixels in size.  We could call pyramid_up()
	// again to find even smaller faces, but note that every time we
	// upsample the image we make the detector run slower since it must
	// process a larger image.
	pyramid_up(img_p);

	// Now tell the face detector to give us a list of bounding boxes
	// around all the faces it can find in the image.
	std::vector<rectangle> dets = detector(img_p);

	if (dets.size() == 0)
	{
		return 0;
	}

	const float scale[2] = { (float)image.x / (float)img_p.nc(), (float)image.y / (float)img_p.nr() };

	rectangleList.resize(dets.size());
	const int dets_sz = dets.size();
#pragma omp parallel for
	for (int i = 0; i < dets_sz; i++)
	{
		rectangle_2d_t rec;

		rec.left   = dets[i].left()*scale[0];
		rec.right  = dets[i].right()*scale[0];
		rec.top    = dets[i].top()*scale[1];
		rec.bottom = dets[i].bottom()*scale[1];

		rectangleList[i] = rec;
	}

	return rectangleList.size();
}



void* create_image_window()
{
	//image_window win;
	return (void*)(new image_window);
}
void close_image_window(void* win_p)
{
	((image_window*)win_p)->close_window();
}

void set_image_window(void*win_p, void* dlib_img)
{
	((image_window*)win_p)->set_size(640, 480);
	((image_window*)win_p)->set_image(*((matrix<bgr_pixel>*)dlib_img));
}

void resize_image_window(void*win_p, int w, int h)
{
	((image_window*)win_p)->set_size(w, h);
}