// Must include our scene first because of GL dependency order
#include "sdfscene.h"

// This will probably already be included by a scene file
#include "glinclude.h"
#include "trackballcamera.h"
// Includes for GLFW
#include <GLFW/glfw3.h>

/// A scene object
SDFScene g_scene;

/// A camera object
//FixedCamera g_camera;
TrackballCamera g_camera;

/******************************************************************
 * GLFW Callbacks
 * These functions are triggered on an event, such as a keypress
 * or mouse click. They need to be passed on to the relevant
 * handler, for example, the camera or scene.
 ******************************************************************/
/**
 * @brief error_callback Function to catch GLFW errors.
 * @param error GLFW error code
 * @param description Text description
 */
void error_callback(int error, const char* description) {
    std::cerr << "Error ("<<error<<"): " << description << "\n";
}

/**
 * @brief cursor_callback Function to catch GLFW cursor movement
 * @param xpos x position
 * @param ypos y position
 */
void cursor_callback(GLFWwindow* /*window*/, double xpos, double ypos) {
    g_camera.handleMouseMove(xpos, ypos);
}

/**
 * @brief mouse_button_callback Handle a mouse click or release
 * @param window Window handle (unused currently)
 * @param button Which button was pressed (e.g. left or right button)
 * @param action GLFW code for the action (GLFW_PRESS or GLFW_RELEASE)
 * @param mods Other keys which are currently being held down (e.g. GLFW_CTRL)
 */
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    // Retrieve the position of the mouse click
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Call the camera to handle the click action
    g_camera.handleMouseClick(xpos, ypos, button, action, mods);
}

/**
 * @brief key_callback Handle key press or release
 * @param window window handle (unused currently)
 * @param key The key that was pressed
 * @param action GLFW code for the action (GLFW_PRESS or GLFW_RELEASE)
 * @param mods Other keys which are currently being held down (e.g. GLFW_CTRL)
 */
void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int mods)
{
    // Escape exits the application
    if (action == GLFW_PRESS) {
        switch(key) {
        case (GLFW_KEY_ESCAPE):
            glfwSetWindowShouldClose(window, true); 
            break;
        case (GLFW_KEY_SPACE):
            g_scene.cycleShapeType();
            break;
        case (GLFW_KEY_B):
            g_scene.toggleBlending();
            break;
        case (GLFW_KEY_1):
            g_scene.setColourMode(1);
            break;
        case (GLFW_KEY_2):
            g_scene.setColourMode(2);
            break;
        case (GLFW_KEY_3):
            g_scene.setColourMode(3);
            break;            
        case (GLFW_KEY_4):
            g_scene.setColourMode(4);  
            break;
        case (GLFW_KEY_5):
            g_scene.setColourMode(5);
            break;
        }
    }
    // Any other keypress should be handled by our camera
    g_camera.handleKey(key, (action == GLFW_PRESS) );
}

/**
 * @brief resize_callback Handle a window resize event.
 * @param width New window width
 * @param height New window height
 */
void resize_callback(GLFWwindow */*window*/, int width, int height) {
    g_camera.resize(width,height);
    g_scene.resizeGL(width,height);
}

int main() {
    if (!glfwInit()) {
        // Initialisation failed
        glfwTerminate();
    }
    
    // Register error callback
    glfwSetErrorCallback(error_callback);

    // Set our OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    // Create our window in a platform agnostic manner
    int width = 1280; int height = 720;
    GLFWwindow* window = glfwCreateWindow(width, // width 
                                          height, // height
                                          "My Title", // title 
                                          nullptr, // monitor for full screen
                                          nullptr); // return value on failure

    if (window == nullptr) {
            // Window or OpenGL context creation failed
            glfwTerminate();
    }
    // Make the window an OpenGL window
    glfwMakeContextCurrent(window);

    // Initialise GLEW - note this generates an "invalid enumerant" error on some platforms
#if ( (!defined(__MACH__)) && (!defined(__APPLE__)) )
    glewExperimental = GL_TRUE;
    glewInit();
    glGetError(); // quietly eat errors from glewInit()
#endif

    // Set keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Disable the cursor for the FPS camera
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse move and click callback
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Set the lastX,Y position on the FPS camera
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    g_camera.setInitialMousePos(mouseX, mouseY);

    // Needed for the fixed camera
    g_camera.setTarget(0.0f, 1.0f, 0.0f);
    g_camera.setEye(0.0f, 10.0f, 15.0f);

    // Initialise our OpenGL scene
    g_scene.initGL();

    // Set the window resize callback and call it once
    glfwSetFramebufferSizeCallback(window, resize_callback);
    resize_callback(window, width, height);

    std::cout << "******************* USAGE **************************\n"
              << "1: Cook-Torrance Shading\n"
              << "2: Visualise Ambient Occlusion\n"
              << "3: Visualise distance on the ground plane\n"
              << "4: Visualise Lambert shading value (e.g. N dot L)\n"
              << "5: Visualise the shadow penumbra\n"
              << "<SPACE>: Cycle between surfaces\n"
              << "b: Toggle blending on/off\n"
              << "****************************************************";

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Poll events
        glfwPollEvents();

        // Update our camera matrices
        g_camera.update();

        // Set the eye position and camera target for the render
        g_scene.setEye(g_camera.getTransformedEye());
        g_scene.setTarget(g_camera.getTarget());
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        g_scene.setMouseX(mouseX);
        g_scene.setMouseY(mouseY);
        // Draw our GL stuff
        g_scene.paintGL();

        // Swap the buffers
        glfwSwapBuffers(window);
    }

    // Close up shop
    glfwDestroyWindow(window);
    glfwTerminate();
}
