#ifndef _SIMPLE_DLIB_UTIL_HPP
#define _SIMPLE_DLIB_UTIL_HPP

#ifdef USE_OPENCV
primitive_image_t cvmat_to_pixcels(cv::Mat& Img)
{
	primitive_image_t p;
	p.pixcels = new unsigned char[Img.rows*Img.cols*Img.channels()];
#pragma omp parallel for
	for (int i = 0; i < Img.rows; i++)
	{
		for (int j = 0; j < Img.cols; j++)
		{
			int pos = (i*Img.cols + j);
			for (int c = 0; c < Img.channels(); c++)
			{
				p.pixcels[Img.channels()*pos + Img.channels() - c - 1] = Img.data[i * Img.step + j * Img.elemSize() + c];
			}
		}
	}
	p.x = Img.cols;
	p.y = Img.rows;
	p.channels = Img.channels();
	return p;
}

cv::Mat pixcels_to_cvmat(primitive_image_t& p)
{
	cv::Mat Img(p.y, p.x, CV_8UC(p.channels));

#pragma omp parallel for
	for (int i = 0; i < Img.rows; i++)
	{
		for (int j = 0; j < Img.cols; j++)
		{
			int pos = (i*Img.cols + j);
			for (int c = 0; c < Img.channels(); c++)
			{
				Img.data[i * Img.step + j * Img.elemSize() + c] = p.pixcels[Img.channels()*pos + Img.channels() - c - 1];
			}
		}
	}
	return Img;
}

#endif

#endif
