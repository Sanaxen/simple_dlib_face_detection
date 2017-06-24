#ifndef _SIMPLE_DLIB_UTIL_HPP
#define _SIMPLE_DLIB_UTIL_HPP

#ifdef USE_OPENCV
primitive_image_t cvmat_to_pixcels(cv::Mat& Img)
{
	primitive_image_t p;
	p.pixcels = new unsigned char[Img.rows*Img.cols*Img.channels()];

	const int sz = Img.rows*Img.cols;
	const int channels = Img.channels();

#pragma omp parallel for
	for (int i = 0; i < sz; i++)
	{
		const cv::Vec3b& pix = Img.at<cv::Vec3b>(i);
		p.pixcels[channels*i + channels - 0 - 1] = pix[0];
		p.pixcels[channels*i + channels - 1 - 1] = pix[1];
		p.pixcels[channels*i + channels - 2 - 1] = pix[2];
	}
	p.x = Img.cols;
	p.y = Img.rows;
	p.channels = channels;
	return p;
}

cv::Mat pixcels_to_cvmat(primitive_image_t& p)
{
	cv::Mat Img(p.y, p.x, CV_8UC(p.channels));

	const int sz = Img.rows*Img.cols;
	const int channels = Img.channels();
#pragma omp parallel for
	for (int i = 0; i < sz; i++)
	{
		cv::Vec3b& pix = Img.at<cv::Vec3b>(i);
		pix[0] = p.pixcels[channels*i + channels - 0 - 1];
		pix[1] = p.pixcels[channels*i + channels - 1 - 1];
		pix[2] = p.pixcels[channels*i + channels - 2 - 1];
	}
	return Img;
}

#endif

#endif
