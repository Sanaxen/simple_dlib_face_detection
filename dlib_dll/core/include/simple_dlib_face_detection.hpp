#ifndef _FACE_DETECTION_EX_HPP
#define _FACE_DETECTION_EX_HPP

#ifdef SIMPLE_DLIB_FACE_DETECTION_EXPORTS
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

DLL_API primitive_image_t primitive_image_clone(primitive_image_t& p);
typedef primitive_image_t(WINAPI *dlib_primitive_image_clone)(primitive_image_t& p);

DLL_API void delete_primitive_image(primitive_image_t& p);
typedef void*(WINAPI *dlib_delete_primitive_image)(primitive_image_t& p);

DLL_API void delete_dlibImg(void* dlibImg);
typedef void*(WINAPI *dlib_delete_dlibImg)(void* dlibImg);


DLL_API int ToBMP(const char* filename, char* newfile);
typedef int(WINAPI *dlib_ToBMP)(const char* filename, char* newfile);

DLL_API void* get_frontal_face_detector_ptr();
typedef void*(WINAPI *dlib_get_frontal_face_detector_ptr)();

DLL_API void load_primitive_image(char* filename, primitive_image_t& p);
typedef void(WINAPI *dlib_load_primitive_image)(char* filename, primitive_image_t& p);

DLL_API void save_primitive_image(char* filename, primitive_image_t p);
typedef void(WINAPI *dlib_save_primitive_image)(char* filename, primitive_image_t p);

DLL_API primitive_image_t dlibImage2primitive_image(void* img_p, const int channels);
typedef primitive_image_t(WINAPI *dlib_dlibImage2primitive_image)(void* img_p, const int channels);

DLL_API void* primitive_image2dlibImage(primitive_image_t& image);
typedef void*(WINAPI *dlib_primitive_image2dlibImage)(primitive_image_t& image);

DLL_API int face_detector(primitive_image_t& image, void* detector_ptr, std::vector<rectangle_2d_t>& rectangleList);
typedef int(WINAPI *dlib_face_detector)(primitive_image_t& image, void* detector_ptr, std::vector<rectangle_2d_t>& rectangleList);

DLL_API void* create_image_window();
typedef void*(WINAPI *dlib_create_image_window)();

DLL_API void close_image_window(void* win_p);
typedef void(WINAPI *dlib_close_image_window)(void* win_p);


DLL_API void set_image_window(void*win_p, void* dlib_img);
typedef void(WINAPI *dlib_set_image_window)(void*win_p, void* dlib_img);

DLL_API void resize_image_window(void*win_p, int w, int h);
typedef void(WINAPI *dlib_resize_image_window)(void*win_p, int w, int h);


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
