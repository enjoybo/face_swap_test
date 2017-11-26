# Target
add_library(pyfaceswap SHARED pyFaceSwap.cpp pyboost_cv3_converter.cpp)
target_include_directories(pyfaceswap PRIVATE 
	${OpenCV_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${Qt5Gui_INCLUDE_DIRS}
)
set_target_properties(pyfaceswap PROPERTIES PREFIX "" OUTPUT_NAME "pyfaceswap")
target_link_libraries(pyfaceswap PRIVATE
    dlib
	face_swap
	${OpenCV_LIBS}
    ${PYTHON_LIBRARY}
	${Boost_LIBRARIES}
	Qt5::Widgets
	Qt5::Gui
)

# Installations
install(TARGETS pyfaceswap LIBRARY DESTINATION lib)
