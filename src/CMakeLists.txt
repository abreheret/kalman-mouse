cmake_minimum_required (VERSION 2.8)
project (KalmanMouse)

macro(find_openCV)
if(WIN32)
	find_path(OpenCV_DIR OpenCVConfig.cmake PATHS 
		     "C:/OpenCV/OpenCV2.4.10/opencv/build" 
		     "C:/your/opencv/build/path"# add your windows OpenCV path here : 
			  )
	if( ${OpenCV_DIR} STREQUAL "OpenCV_DIR-NOTFOUND" ) 
		message( FATAL_ERROR "FATAL_ERROR : OpenCV is not found, Please insert your OpenCV Path")
	else ()
		message(STATUS "Your OpenCV_DIR is ${OpenCV_DIR}")
	endif()
	find_package(OpenCV REQUIRED)
else()
	find_package(OpenCV REQUIRED)
endif()
endmacro()

find_openCV()
add_executable(KalmanMouse main.cpp)
target_link_libraries( KalmanMouse ${OpenCV_LIBS})
