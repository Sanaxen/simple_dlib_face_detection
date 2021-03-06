#ifndef _FACE_DETECTION_EX_DLL_HPP
#define _FACE_DETECTION_EX_DLL_HPP

#ifdef SIMPLE_FACE_DETECTION_EXPORTS
#define DLL_API	__declspec(dllexport) 
#else
#define DLL_API /* */
#endif

#include <Windows.h>
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif
HMODULE __hModule = NULL;

typedef struct primitive_image
{
	int x;
	int y;
	int channels;
	unsigned char* pixcels;

} primitive_image_t;

typedef struct rectangle_2d
{
	float left;
	float right;
	float top;
	float bottom;

}rectangle_2d_t;

typedef primitive_image_t(WINAPI *dlib_primitive_image_clone)(primitive_image_t& p);

typedef void*(WINAPI *dlib_delete_primitive_image)(primitive_image_t& p);

typedef void*(WINAPI *dlib_delete_dlibImg)(void* dlibImg);


typedef int(WINAPI *dlib_ToBMP)(const char* filename, char* newfile);

typedef void*(WINAPI *dlib_get_frontal_face_detector_ptr)();

typedef void(WINAPI *dlib_load_primitive_image)(char* filename, primitive_image_t& p);

typedef void(WINAPI *dlib_save_primitive_image)(char* filename, primitive_image_t p);

typedef primitive_image_t(WINAPI *dlib_dlibImage2primitive_image)(void* img_p, const int channels);

typedef void*(WINAPI *dlib_primitive_image2dlibImage)(primitive_image_t& image);

typedef int(WINAPI *dlib_face_detector)(void* win_p, primitive_image_t& image, void* detector_ptr, std::vector<rectangle_2d_t>& rectangleList);

typedef int(WINAPI *dlib_face_detector_cv)(void* win_p, void* cvMat_image, void* detector_ptr, std::vector<rectangle_2d_t>& rectangleList, void* shape_predictor_model);

typedef void*(WINAPI *dlib_create_image_window)();

typedef void(WINAPI *dlib_close_image_window)(void* win_p);


typedef void(WINAPI *dlib_set_image_window)(void*win_p, void* dlib_img);

typedef void(WINAPI *dlib_resize_image_window)(void*win_p, int w, int h);

typedef void*(WINAPI *dlib_new_shape_predictor)(char* shape_predictor_data);

typedef void(WINAPI *dlib_delete_shape_predictor)(void* model);

typedef void(WINAPI *dlib_delete_full_object_detection_shape)(void* shape);


#define DNN_DEF_FUNC(f)	dlib_ ## f f ## _dlib = NULL;
#define DNN_FUNC(f)	f ## _dlib = (dlib_ ## f)GetProcAddress(__hModule, # f);if ( f ## _dlib == NULL ) printf("load %s error.\n", #f);


DNN_DEF_FUNC(primitive_image_clone);
DNN_DEF_FUNC(delete_primitive_image);
DNN_DEF_FUNC(delete_dlibImg);
DNN_DEF_FUNC(ToBMP);
DNN_DEF_FUNC(get_frontal_face_detector_ptr);
DNN_DEF_FUNC(load_primitive_image);
DNN_DEF_FUNC(save_primitive_image);
DNN_DEF_FUNC(dlibImage2primitive_image);
DNN_DEF_FUNC(primitive_image2dlibImage);
DNN_DEF_FUNC(face_detector);
DNN_DEF_FUNC(face_detector_cv);
DNN_DEF_FUNC(new_shape_predictor);
DNN_DEF_FUNC(delete_shape_predictor);
DNN_DEF_FUNC(delete_full_object_detection_shape);
DNN_DEF_FUNC(create_image_window);
DNN_DEF_FUNC(close_image_window);
DNN_DEF_FUNC(set_image_window);
DNN_DEF_FUNC(resize_image_window);

inline int simple_dlib_init(const char* this_dll)
{
	// DLLのロード
	__hModule = LoadLibraryA(this_dll);
	//if (__hModule == NULL)
	//{
	//  printf("%s", "DLLのロードに失敗しました。");
	//  return -1;
	//}

	// 関数のアドレス取得
	//dn_CreateNET CreateNET = (dn_CreateNET)GetProcAddress(hModule, "CreateNET");
	//if (CreateNET == NULL)
	//{
	//  printf("%s", "関数のアドレス取得に失敗しました。");
	//  FreeLibrary(hModule);
	//  return 0;
	//}
	DNN_FUNC(primitive_image_clone);
	DNN_FUNC(delete_primitive_image);
	DNN_FUNC(delete_dlibImg);
	DNN_FUNC(ToBMP);
	DNN_FUNC(get_frontal_face_detector_ptr);
	DNN_FUNC(load_primitive_image);
	DNN_FUNC(save_primitive_image);
	DNN_FUNC(dlibImage2primitive_image);
	DNN_FUNC(primitive_image2dlibImage);
	DNN_FUNC(face_detector);
	DNN_FUNC(face_detector_cv);
	DNN_FUNC(new_shape_predictor);
	DNN_FUNC(delete_shape_predictor);
	DNN_FUNC(delete_full_object_detection_shape);
	DNN_FUNC(create_image_window);
	DNN_FUNC(close_image_window);
	DNN_FUNC(set_image_window);
	DNN_FUNC(resize_image_window);
	return 0;
}

inline void simple_dlib_term()
{
	FreeLibrary(__hModule);
}

#ifdef __cplusplus
};
#endif

#endif
