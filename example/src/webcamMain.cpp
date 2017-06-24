#include <opencv2/opencv.hpp>
#include "../../include/simple_dlib_face_detection_dll.hpp"

#define USE_OPENCV
#include "../../include/util/simple_dlib_face_detection_util.hpp"

#define USE_OPENCV_WINDOW	0
int main(int argh, char* argv[])
{
	simple_dlib_init("simple_dlib_face_detection.dll");

	int device = 0;
	if (argh == 2) device = atoi(argv[1]);

	cv::VideoCapture cap(0);//デバイスのオープン
							//cap.open(0);//こっちでも良い．

	if (!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
	{
		//読み込みに失敗したときの処理
		return -1;
	}

#if USE_OPENCV_WINDOW
	void* win = 0;
#else
	void* win = create_image_window_dlib();
	resize_image_window_dlib(win, 640, 480);
#endif

#if USE_OPENCV_WINDOW
	cv::Size cap_size(640*1.2, 480*1.2);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, cap_size.width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, cap_size.height);
	//cap.set(CV_CAP_PROP_FPS, 30);
#endif

	void* frontal_face_detector_ptr = get_frontal_face_detector_ptr_dlib();

	void* shape_predictor_model = 0;
	const char* model = "shape_predictor_68_face_landmarks.dat";
	FILE* fp = fopen( model, "r") ;
	if ( fp == NULL )
	{
		printf("\"shape_predictor_68_face_landmarks.dat\" not found.\n");
		printf("download->http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n");
	}else
	{
		fclose(fp);
		shape_predictor_model = new_shape_predictor_dlib( (char*)model);
	}


	while (1)//無限ループ
	{
		cv::Mat frame;
		cap >> frame; // get a new frame from camera

		std::vector<rectangle_2d_t> rectList;

#if 10
		const int faceNum = face_detector_cv_dlib(win, (void*)&frame, (void*)frontal_face_detector_ptr, rectList, shape_predictor_model);
#else
		primitive_image_t& frame_img = cvmat_to_pixcels(frame);

		const int faceNum = face_detector_dlib(win, frame_img, (void*)frontal_face_detector_ptr, rectList);
		delete[] frame_img.pixcels;
#endif

		if ( win ) resize_image_window_dlib(win, 640, 480);
		if (faceNum)
		{
			const int rectList_sz = rectList.size();
#pragma omp parallel for
			for (int j = 0; j < rectList_sz; ++j)
			{
				cv::Point p1((int)rectList[j].left, (int)rectList[j].top);
				cv::Point p2((int)rectList[j].right, (int)rectList[j].bottom);
				try
				{
					cv::rectangle(frame, p1, p2, cv::Scalar(0, 0, 255), 1, CV_AA);
				}
				catch (...)
				{
					continue;
				}
			}
		}


		if (!win)
		{
			cv::imshow("window", frame);//画像を表示．

			int key = cv::waitKey(1);
			if (key == 113)//qボタンが押されたとき
			{
				break;//whileループから抜ける．
			}
			else if (key == 115)//sが押されたとき
			{
				//フレーム画像を保存する．
				//cv::imwrite("img.png", frame);
			}

		}

	}
	cv::destroyAllWindows();
	return 0;
}
