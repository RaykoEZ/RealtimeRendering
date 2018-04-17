#include "NGLScene.h"
#include <QGuiApplication>
#include <QMouseEvent>
//#include <ngl/Camera.h>
//#include <ngl/Light.h>
#include <ngl/Material.h>
#include <ngl/NGLInit.h>
#include <ngl/NGLStream.h>
#include <ngl/ShaderLib.h>
#include <ngl/VAOPrimitives.h>
//#include <ngl/Image.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext.hpp>
NGLScene::NGLScene()
{
  setTitle( "Qt5 Simple NGL Demo" );
  m_startTime = std::chrono::high_resolution_clock::now();
  m_colourMode = 1;
  m_shapeType = 0;
  m_isBlending = true;
  m_eye = glm::vec3(0.0, 0.0, 2.0);
  m_lookAt = glm::vec3(0.0, 0.0, 0.0);


}


NGLScene::~NGLScene()
{
  std::cout << "Shutting down NGL, removing VAO's and Shaders\n";
}



void NGLScene::resizeGL( int _w, int _h )
{
  //m_cam.setShape( 45.0f, static_cast<float>( _w ) / _h, 0.05f, 350.0f );
  m_win.width  = static_cast<int>( _w * devicePixelRatio() );
  m_win.height = static_cast<int>( _h * devicePixelRatio() );
}


void NGLScene::initializeGL()
{
  //auto mouseX = cursor().pos().x();
  //auto mouseY = cursor().pos().y();
  //m_cam.setInitialMousePos(float(mouseX),float(mouseY));
  // we must call that first before any other GL commands to load and link the
  // gl commands from the lib, if that is not done program will crash
  ngl::NGLInit::instance();
  // Set background colour
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  // enable depth testing for drawing
  glEnable( GL_DEPTH_TEST );
// enable multisampling for smoother drawing

  glEnable( GL_MULTISAMPLE );

  // now to load the shader and set the values
  // grab an instance of shader manager
  constexpr auto shaderProgram = "Marbles";
  //constexpr auto vertexShader  = "MarblesVertex";
  //constexpr auto fragShader    = "MarblesFragment";
  ngl::ShaderLib* shader = ngl::ShaderLib::instance();
  shader->loadShader(shaderProgram,"shaders/sdf_vert.glsl","shaders/sdf_frag.glsl");
  ngl::VAOPrimitives *prim=ngl::VAOPrimitives::instance();
  prim->createTrianglePlane("plane",2,2,1,1,ngl::Vec3(0,1,0));


  // we are creating a shader called Phong to save typos
  // in the code create some constexpr
  /*
  // create the shader program
  shader->createShaderProgram( shaderProgram );
  // now we are going to create empty shaders for Frag and Vert
  shader->attachShader( vertexShader, ngl::ShaderType::VERTEX );
  shader->attachShader( fragShader, ngl::ShaderType::FRAGMENT );
  // attach the source
  //shader->loadShaderSource( vertexShader, "shaders/MarblesVert.glsl" );
  //shader->loadShaderSource( fragShader, "shaders/MarblesFragment.glsl" );
  shader->loadShaderSource(vertexShader,"shaders/MarblesVertex.glsl");
  shader->loadShaderSource(fragShader,"shaders/MarblesFragment.glsl");
  // compile the shaders
  shader->compileShader( vertexShader );
  shader->compileShader( fragShader );
  // add them to the program
  shader->attachShaderToProgram( shaderProgram, vertexShader );
  shader->attachShaderToProgram( shaderProgram, fragShader );
  */
  //initEnvironment();
  //initTexture(1, m_glossMapTex, "images/gloss.png");

  // now we have associated that data we can link the shader
  //shader->linkProgramObject( shaderProgram );
  // and make it active ready to load values
  //( *shader )[ shaderProgram ]->use();
  //shader->setUniform("resolution",ngl::Vec3(float(m_width),float(m_height),0.0f));
  //shader->setUniform("lookAt",m_lookAt);
  //shader->setUniform("eyePos",m_eye);
  //shader->setUniform("time",0.0f);
  //shader->setUniform("glossMap", 1);
  // the shader will use the currently active material and light0 so set them
  //ngl::Material m( ngl::STDMAT::GOLD );
  // load our material values to the shader into the structure material (see Vertex shader)
  //m.loadToShader( "material" );
  // Now we will create a basic Camera from the graphics library
  // This is a static camera so it only needs to be set once
  // First create Values for the camera position
  //ngl::Vec3 from( 0.0,0.0,1.0 );
  //ngl::Vec3 to( 0, 0, 0 );
  //ngl::Vec3 up( 0, 1, 0 );
  // now load to our new camera
  //m_cam.set( from, to, up );
  // set the shape using FOV 45 Aspect Ratio based on Width and Height
  // The final two are near and far clipping planes of 0.5 and 10
  //m_cam.setShape( 45.0f, 720.0f / 576.0f, 0.05f, 350.0f );
  //shader->setUniform( "viewerPos", m_cam.getEye().toVec3() );
  // now create our light that is done after the camera so we can pass the
  // transpose of the projection matrix to the light to do correct eye space
  // transformations
  //ngl::Mat4 iv = m_cam.getViewMatrix();

  //iv.transpose();
  //ngl::Light light( ngl::Vec3( -2, 5, 2 ), ngl::Colour( 1, 1, 1, 1 ), ngl::Colour( 1, 1, 1, 1 ), ngl::LightModes::POINTLIGHT );
  //light.setTransform( iv );
  //m_screenQuad.reset( new ScreenQuad("Marbles"));
  //glViewport(0,0,width(),height());
  // load these values to the shader as well
  //light.loadToShader( "light" );

  //ngl::VAOPrimitives *prim=ngl::VAOPrimitives::instance();
  //prim->createTrianglePlane("plane",10,10,5,5,ngl::Vec3(0,1,0));
  //startTimer(10);
}


void NGLScene::loadMatricesToShader()
{
/*
  ngl::ShaderLib* shader = ngl::ShaderLib::instance();
  //shader->setUniform("lookAt",ngl::Vec3(m_win.spinXFace,m_win.spinYFace, 0));
  //shader->setUniform("eyePos",ngl::Vec3(0.0,0.0,4.0));
  //shader->setUniform("mouse",ngl::Vec2(m_win.spinXFace,m_win.spinYFace));
  //shader->setUniform("cam_pos",ngl::Vec3(0.0,0.0,15.0));
  //shader->setUniform("resolution",ngl::Vec2(float(width()),float(height())));

  //GLint pid = shader->getProgramID("Marbles");
  // Calculate the elapsed time since the programme started
  auto now = std::chrono::high_resolution_clock::now();
  double t = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_startTime).count()  * 0.001;
  //ngl::Mat4 MV;
  //ngl::Mat4 MVP;
  //ngl::Mat3 N;
  //ngl::Mat4 M;
  //M            = m_mouseGlobalTX;
  //MV           = m_cam.getViewMatrix() * M;
  //MVP          = m_cam.getVPMatrix() * M;
  //alligning plane to screen
  //MVP.rotateZ(90.0);

  //N = MV;
  //N.inverse().transpose();
  //shader->setUniform( "MV", MV );
  //shader->setUniform( "MVP", MVP );
  //shader->setUniform( "N", N );
  //shader->setUniform( "M", M );
  //For sdf, we need these uniforms
  // Set the viewport resolution
  //glUniform3fv(glGetUniformLocation(pid, "resolution"), 1, glm::value_ptr(glm::vec3(float(m_width), float(m_height), 0.0f)));

  // Set the time elapsed since the programme started
  //glUniform1f(glGetUniformLocation(pid, "time"), float(t));

  // The default NGL plane isn't screen oriented so we still have to rotate it around the x-axis
  // to align with the screen
  //shader->setUniform("resolution",ngl::Vec3(float(m_win.width),float(m_win.height),0.0f));
  //shader->setUniform("eyePos",ngl::Vec3(m_eye));
  //shader->setUniform("lookAt",ngl::Vec3(m_lookAt));
  // Transfer over the eye and target position
  //glUniform3fv(glGetUniformLocation(pid, "eyePos"), 1, glm::value_ptr(m_eye));
  //glUniform3fv(glGetUniformLocation(pid, "lookAt"), 1, glm::value_ptr(m_lookAt));
  */
}

void NGLScene::paintGL()
{
  m_cam.updateMe();
  m_eye = m_cam.getTransformedEye();
  m_lookAt = m_cam.getTarget();
  // Clear the screen (fill with our glClearColor)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set up the viewport
  //glViewport(0,0,m_width,m_height);
  glViewport( 0, 0, width(), height() );
  // Use our shader for this draw
  ngl::ShaderLib *shader=ngl::ShaderLib::instance();
  (*shader)["Marbles"]->use();
  GLint pid = shader->getProgramID("Marbles");

  // Our MVP matrices
  glm::mat4 M = glm::mat4(1.0f);
  glm::mat4 MVP, MV;
  glm::mat3 N;

  // Note the matrix multiplication order as we are in COLUMN MAJOR storage
  MV = m_cam.viewMatrix() * M;
  N = glm::inverse(glm::mat3(MV));
  MVP = m_cam.projMatrix() * MV;

  // Calculate the elapsed time since the programme started
  auto now = std::chrono::high_resolution_clock::now();
  double t = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_startTime).count()  * 0.001;

  // Set the viewport resolution
  glUniform3fv(glGetUniformLocation(pid, "iResolution"), 1, glm::value_ptr(glm::vec3(float(m_width), float(m_height), 0.0f)));

  // Set the time elapsed since the programme started
  glUniform1f(glGetUniformLocation(pid, "iTime"), float(t));

  // Set the current colour mode (0,1 or 2)
  glUniform1i(glGetUniformLocation(pid, "colourMode"), m_colourMode);

  // Set the current shape to render
  glUniform1i(glGetUniformLocation(pid, "shapeType"), m_shapeType);

  // Set whether blending is used between shapes
  glUniform1i(glGetUniformLocation(pid, "isBlending"), (m_isBlending)?1:0);

  // The default NGL plane isn't screen oriented so we still have to rotate it around the x-axis
  // to align with the screen
  MVP = glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 0.5f, glm::vec3(1.0f,0.0f,0.0f));
  glUniformMatrix4fv(glGetUniformLocation(pid, "MVP"), 1, false, glm::value_ptr(MVP));

  // Transfer over the eye and target position
  glUniform3fv(glGetUniformLocation(pid, "eyepos"), 1, glm::value_ptr(m_eye));
  glUniform3fv(glGetUniformLocation(pid, "target"), 1, glm::value_ptr(m_lookAt));

  // Draw the plane that we've created
  ngl::VAOPrimitives *prim=ngl::VAOPrimitives::instance();
  prim->draw("plane");
  //glViewport( 0, 0, m_win.width, m_win.height );
  // clear the screen and depth buffer
  //glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  // grab an instance of the shader manager
  //ngl::ShaderLib* shader = ngl::ShaderLib::instance();
  //( *shader )[ "Marbles" ]->use();

  // Rotation based on the mouse position for our global transform
  //ngl::Mat4 rotX;
  //ngl::Mat4 rotY;
  // create the rotation matrices
  //rotX.rotateX( m_win.spinXFace );
  //rotY.rotateY( m_win.spinYFace );
  // multiply the rotations
  //m_mouseGlobalTX = rotX * rotY;
  // add the translations
  //m_mouseGlobalTX.m_m[ 3 ][ 0 ] = m_modelPos.m_x;
  //m_mouseGlobalTX.m_m[ 3 ][ 1 ] = m_modelPos.m_y;
  //m_mouseGlobalTX.m_m[ 3 ][ 2 ] = m_modelPos.m_z;


  //loadMatricesToShader();
  // get the VBO instance and draw the built in teapot
  //ngl::VAOPrimitives* prim = ngl::VAOPrimitives::instance();
  //prim->draw( "teapot" );
  //m_screenQuad->draw();
  //m_mesh->draw();
}

//----------------------------------------------------------------------------------------------------------------------

void NGLScene::keyPressEvent( QKeyEvent* _event )
{
  // that method is called every time the main window recives a key event.
  // we then switch on the key value and set the camera in the GLWindow
    switch (_event->key())
    {
    // escape key to quite
    case Qt::Key_Escape : QGuiApplication::exit(EXIT_SUCCESS); break;
    // turn on wirframe rendering
    case Qt::Key_W : glPolygonMode(GL_FRONT_AND_BACK,GL_LINE); break;
    // turn off wire frame
    case Qt::Key_S : glPolygonMode(GL_FRONT_AND_BACK,GL_FILL); break;
    // show full screen
    case Qt::Key_F : showFullScreen(); break;
    // show windowed
    case Qt::Key_N : showNormal(); break;
    default : break;
    }
    // finally update the GLWindow and re-draw
    //if (isExposed())
      update();
}

void NGLScene::timerEvent(QTimerEvent *_event)
{
  //static float t=0.0;
  // grab an instance of the shader manager
  //ngl::ShaderLib *shader=ngl::ShaderLib::instance();
  //(*shader)["Marbles"]->use();
  //shader->setUniform("time",t);
  //t+=0.01;
  //if(t > 5.0) t=0.0;
  //update();
}


