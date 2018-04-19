# These are common compilation options shared with all workshops

# The application type
TEMPLATE = app

# Is this using the lab build? (this is for my personal use - RS)
#LAB_BUILD = false
LAB_BUILD = true
# Check to see if you have a custom NGL installation directory
NGLPATH = $$(NGLDIR)
isEmpty(NGLPATH) {
  NGLPATH = $$(HOME)/NGL
} 

# Check to see if there is a custom IGL directory (I've put it on /public/devel)
IGLPATH = $$(IGLDIR)
isEmpty(IGLPATH) {
  IGLPATH = /public/devel/libigl
}

# Check to see if there is a custom libnoise directory (this is part of the standard build)
TEST = $$(NOISEDIR)
isEmpty(TEST) {
  NOISEINCPATH = /public/devel/noise/src
  NOISELIBPATH = /public/devel/noise/lib
} 
else {
  NOISEINCPATH = $$(NOISEDIR)/src
  NOISELIBPATH = $$(NOISEDIR)/lib
}

# Check to see if there is a custom Eigen installation
EIGENPATH = $$(EIGENDIR)
isEmpty(EIGENPATH) {
  EIGENPATH = /usr/local/include/eigen3
}

# The include path - includes a number of environment variables (see README.md)
INCLUDEPATH += $$NGLPATH/include ../common/include $$IGLPATH/include $$EIGENPATH $$NOISEINCPATH

# The config includes C++11 features. I'll assume you want debug mode!
CONFIG += c++11 debug

# These options are to include the openGL headers etc for NGL
QT += core opengl gui

# The path to move all your compiled object files to
OBJECTS_DIR = obj

# The libs path (this will change depending on your platform
if ($$LAB_BUILD) {
  LIBS += -lXi -ldl -lX11 -lXrandr -lXinerama -lXcursor -lXxf86vm -lglfw3
} else {
  !win32:LIBS += -lglfw
  win32:
    {

    #these paths are to be changed for your PC's lib/include paths
    INCLUDEPATH += "c:/glew/include"
    INCLUDEPATH += "c:/glm"
    INCLUDEPATH += "c:/glfw/include"
    DEPENDPATH += "c:/glfw/include"
    LIBS += -LC:/glfw/lib/ -lglfw3dll
    #using glfw's dll/shared lib needs this defined here
    DEFINES+=GLFW_DLL
    }
}
#LIBS += -lGL -lGLEW -lGLU $$NOISELIBPATH/libnoise.a

# The RPATH tells the executable where to find the shared libraries that are not in LD_LIBRARY_PATH
QMAKE_RPATHDIR += $$NGLPATH/lib 

# Include the NGL pri file
include($$NGLPATH/UseNGL.pri)
