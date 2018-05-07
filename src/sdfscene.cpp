#include "sdfscene.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/ext.hpp>
// 3rd party library for loading hdr files
#include <ngl/Obj.h>
#include <ngl/NGLInit.h>
#include <ngl/VAOPrimitives.h>
#include <ngl/ShaderLib.h>
#include <NGL/Image.h>

SDFScene::SDFScene() : Scene() {
    // Set the time since we started running the scene
    m_startTime = std::chrono::high_resolution_clock::now();
    m_colourMode = 1;
    //m_shapeType = 0;
    m_eye = glm::vec3(0.0, 0.0, 2.0);
    m_target = glm::vec3(0.0, 0.0, 0.0);
}

/**
 * @brief ObjLoaderScene::initGL
 */
void SDFScene::initGL() noexcept {
    // Fire up the NGL machinary (not doing this will make it crash)
    ngl::NGLInit::instance();

    // Set background colour
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // enable depth testing for drawing
    glEnable(GL_DEPTH_TEST);

    // enable multisampling for smoother drawing
    glEnable(GL_MULTISAMPLE);

    // Create and compile all of our shaders
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    shader->loadShader("SDFProgram","shaders/sdf_vert.glsl","shaders/sdf_frag.glsl");

    initTexture(0, m_glossTex, "images/glossValue_alt1.jpg");
    // Set the active texture unit on the GPU
    GLint pid = shader->getProgramID("SDFProgram");
    glUniform1i(glGetUniformLocation(pid, "GlossTexture"), //location of uniform
                       0); // texture unit for colour

    // Create a screen oriented plane
    ngl::VAOPrimitives *prim=ngl::VAOPrimitives::instance();
    prim->createTrianglePlane("plane",2,2,1,1,ngl::Vec3(0,1,0));
}

void SDFScene::paintGL() noexcept {
    // Clear the screen (fill with our glClearColor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up the viewport
    glViewport(0,0,m_width,m_height);

    // Use our shader for this draw
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    (*shader)["SDFProgram"]->use();
    GLint pid = shader->getProgramID("SDFProgram");

    // Our MVP matrices
    glm::mat4 M = glm::mat4(1.0f);
    glm::mat4 MVP, MV;
    glm::mat3 N;

    // Note the matrix multiplication order as we are in COLUMN MAJOR storage
    MV = m_V * M;
    N = glm::inverse(glm::mat3(MV));
    MVP = m_P * MV;

    // Calculate the elapsed time since the programme started
    auto now = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_startTime).count()  * 0.001;

    // Set the viewport resolution
    glUniform3fv(glGetUniformLocation(pid, "iResolution"), 1, glm::value_ptr(glm::vec3(float(m_width), float(m_height), 0.0f)));

    // Set the time elapsed since the programme started
    glUniform1f(glGetUniformLocation(pid, "iTime"), float(t));

    // Set the current colour mode (0,1 or 2)
    glUniform1i(glGetUniformLocation(pid, "colourMode"), m_colourMode);
    //glUniform3fv(glGetUniformLocation(pid, "iMouse"), 1, glm::value_ptr(glm::vec3(float(m_mouseX),float(m_mouseY),0.0f)));
    // Set the current shape to render
    //glUniform1i(glGetUniformLocation(pid, "shapeType"), m_shapeType);
    
    // The default NGL plane isn't screen oriented so we still have to rotate it around the x-axis
    // to align with the screen
    MVP = glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 0.5f, glm::vec3(1.0f,0.0f,0.0f));

    //MV = glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 0.5f, glm::vec3(1.0f,0.0f,0.0f));
    glUniformMatrix4fv(glGetUniformLocation(pid, "MVP"), 1, false, glm::value_ptr(MVP));
    glUniform3fv(glGetUniformLocation(pid, "eyepos"), 1, glm::value_ptr(m_eye));
    glUniform3fv(glGetUniformLocation(pid, "target"), 1, glm::value_ptr(m_target));
    
    // Draw the plane that we've created
    ngl::VAOPrimitives *prim=ngl::VAOPrimitives::instance();
    prim->draw("plane");    
}



void SDFScene::setColourMode(const int& _mode) {    
    if ((_mode >= 1) && (_mode <= 5)) {
        m_colourMode = _mode;
    }
}
void SDFScene::initTexture(const GLint& texUnit, GLuint &texId, const char *filename) {
    // Set our active texture unit
    glActiveTexture(GL_TEXTURE0 + texUnit);

    // Load up the image using NGL routine
    ngl::Image img(filename);

    // Create storage for our new texture
    glGenTextures(1, &texId);

    // Bind the current texture
    glBindTexture(GL_TEXTURE_2D, texId);

    // Transfer image data onto the GPU using the teximage2D call
    glTexImage2D (
                GL_TEXTURE_2D,    // The target (in this case, which side of the cube)
                0,                // Level of mipmap to load
                img.format(),     // Internal format (number of colour components)
                img.width(),      // Width in pixels
                img.height(),     // Height in pixels
                0,                // Border
                GL_RGB,          // Format of the pixel data
                GL_UNSIGNED_BYTE, // Data type of pixel data
                img.getPixels()); // Pointer to image data in memory

    // Set up parameters for our texture
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

/**
 * @brief Scene::initEnvironment in the specified texture unit
 */
void SDFScene::initEnvironment(const GLint& texUnit) {
    // Enable seamless cube mapping
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // Placing our environment map texture in texture unit 0
    glActiveTexture(GL_TEXTURE0 + texUnit);

    // Generate storage and a reference for our environment map texture
    glGenTextures (1, &m_envTex);

    // Bind this texture to the active texture unit
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_envTex);

    // Now load up the sides of the cube
    initEnvironmentSide(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, "images/sky_zneg.png");
    initEnvironmentSide(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, "images/sky_zpos.png");
    initEnvironmentSide(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, "images/sky_ypos.png");
    initEnvironmentSide(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, "images/sky_yneg.png");
    initEnvironmentSide(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, "images/sky_xneg.png");
    initEnvironmentSide(GL_TEXTURE_CUBE_MAP_POSITIVE_X, "images/sky_xpos.png");

    // Generate mipmap levels
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    // Set the texture parameters for the cube map
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GLfloat anisotropy;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropy);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);

    // Set our cube map texture to on the shader so we can use it
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    shader->use("SDFProgram");
    shader->setUniform("EnvironmentTexture", texUnit);
}

/**
 * @brief Scene::initEnvironmentSide
 * @param texture
 * @param target
 * @param filename
 * This function should only be called when we have the environment texture bound already
 * copy image data into 'target' side of cube map
 */
void SDFScene::initEnvironmentSide(GLenum target, const char *filename) {
    // Load up the image using NGL routine
    ngl::Image img(filename);

    // Transfer image data onto the GPU using the teximage2D call
    glTexImage2D (
      target,           // The target (in this case, which side of the cube)
      0,                // Level of mipmap to load
      img.format(),     // Internal format (number of colour components)
      img.width(),      // Width in pixels
      img.height(),     // Height in pixels
      0,                // Border
      GL_RGBA,          // Format of the pixel data
      GL_UNSIGNED_BYTE, // Data type of pixel data
      img.getPixels()   // Pointer to image data in memory
    );
}
