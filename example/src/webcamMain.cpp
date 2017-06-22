#include <opencv2/opencv.hpp>
#include "../../include/simple_dlib_face_detection_dll.hpp"

#define USE_OPENCV
#include "../../include/util/simple_dlib_face_detection_util.hpp"

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

	cv::Size cap_size(640*1.2, 480*1.2);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, cap_size.width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, cap_size.height);
	//cap.set(CV_CAP_PROP_FPS, 30);

	void* frontal_face_detector_ptr = get_frontal_face_detector_ptr_dlib();

#if 0
#if 0
	cv::CascadeClassifier cascade("C:\\dev\opencv-3.1.0\\sources\\data\\hogcascades\\hogcascade_pedestrians.xml");
#else	
	cv::HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
#endif
#endif

	while (1)//無限ループ
	{
		cv::Mat frame;
		cap >> frame; // get a new frame from camera

					  //
					  //取得したフレーム画像に対して，クレースケール変換や2値化などの処理を書き込む．
					  //

		std::vector<rectangle_2d_t> rectList;

		primitive_image_t& frame_img = cvmat_to_pixcels(frame);

		int faceNum = face_detector_dlib(frame_img, (void*)frontal_face_detector_ptr, rectList);
		delete [] frame_img.pixcels;

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

		std::vector<cv::Rect> people;
#if 0
#if 0
		cascade.detectMultiScale(frame, people, 1.1, 2);
#else
		hog.detectMultiScale(frame, people, 0, cv::Size(8, 8), cv::Size(16, 16), 1.05, 2);
#endif
#endif

		for (int j = 0; j < people.size(); j++)
		{
			cv::rectangle(frame, people[j].tl(), people[j].br(), cv::Scalar(255, 0, 0), 1, CV_AA);

		}
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
	cv::destroyAllWindows();
	return 0;
}
