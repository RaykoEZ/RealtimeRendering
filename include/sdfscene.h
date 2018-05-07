#ifndef SDFSCENE_H
#define SDFSCENE_H

// The parent class for this scene
#include "scene.h"
#include <chrono>
#include <ngl/Obj.h>

class SDFScene : public Scene
{
public:
    /// Constructor
    SDFScene();

    /// Called when the scene needs to be painted
    void paintGL() noexcept;

    /// Called when the scene is to be initialised
    void initGL() noexcept;

    /// Set the method used to colour the scene
    void setColourMode(const int&);

    /// Cycle around the shape types (currently 6 are supported)
    void cycleShapeType() {m_shapeType = (m_shapeType + 1) % 6;}

    /// Toggle whether blending is used
    void toggleBlending() {m_isBlending = !m_isBlending;}

    /// Set the target and eye position
    void setEye(const glm::vec3& eye) {m_eye = eye;}
    void setTarget(const glm::vec3& target) {m_target = target;}
    void setMouseX(const double &_x){ m_mouseX = _x;}
    void setMouseY(const double &_y){ m_mouseY = _y;}
private:

    GLuint m_glossTex,m_envTex;
    /// Keep track of the last time
    std::chrono::high_resolution_clock::time_point m_startTime;

    /// A user switch to change the way colour is used in the shader and which shapes to draw
    int m_colourMode, m_shapeType;

    /// Determine whether blending is used in the shader
    bool m_isBlending;
    double m_mouseX = 0.0, m_mouseY = 0.0;
    /// Sets the target and eye position on the shader
    glm::vec3 m_eye, m_target;

    /// Initialise a texture with the given texUnit
    void initTexture(const GLint &texUnit, GLuint &texId, const char *filename);

    /// Initialise our Environment Texture Maps
    void initEnvironment(const GLint& texUnit);

    /// Initialise a single side of our environment map
    void initEnvironmentSide(GLenum target, const char *filename);


};

#endif // SDFSCENE_H
