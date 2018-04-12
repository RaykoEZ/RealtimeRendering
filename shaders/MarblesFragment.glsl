#version 420 core

// Attributes passed on from the vertex shader
//smooth in vec3 FragmentPosition;
//smooth in vec3 FragmentNormal;
smooth in vec2 FragmentTexCoord;

/// @brief our output fragment colour
layout (location=0) out vec4 FragColour;

// A texture unit for storing the 3D texture
//uniform samplerCube envMap;

// Set the maximum environment level of detail (cannot be queried from GLSL apparently)
//uniform int envMaxLOD = 8;

// Set our gloss map texture
//uniform sampler2D glossMap;

// The inverse View matrix
//uniform mat4 invV;

// A toggle allowing you to set it to reflect or refract the light
//uniform bool isReflect = false;

// Specify the refractive index for refractions
//uniform float refractiveIndex = 1.0;

const float tau = 6.283185;              // A modeling constant
// Structure for holding light parameters
struct LightInfo
{
    vec4 Position; // Light position in eye coords.
    vec3 La; // Ambient light intensity
    vec3 Ld; // Diffuse light intensity
    vec3 Ls; // Specular light intensity
};

// We'll have a single light in the scene with some default values
uniform LightInfo Light = LightInfo(
            vec4(0.0, 10.0, 10.0, 1.0),   // position
            vec3(0.2, 0.2, 0.2),        // La
            vec3(1.0, 1.0, 1.0),        // Ld
            vec3(1.0, 1.0, 1.0)         // Ls
            );


// Ambient occlusion parameters
uniform int aoIter = 8;                  // The number of iterations for ambient occlusion. Higher is better quality.
uniform float aoDist = 1.0;              // The size of the ambient occlusion kernel
uniform float aoPower = 16.0;             // The exponent for the ao kernel - a larger power increases the fall-off

// set important material values
uniform float roughnessValue = 0.1; // 0 : smooth, 1: rough
uniform float F0 = 0.5; // fresnel reflectance at normal incidence
uniform float k = 0.5;  // fraction of diffuse reflection (specular reflection = 1 - k)


//--------------raymarching uniforms---------------
uniform int marchIter = 256;             // The maximum number of iterations for the ray marching algorithm
uniform int shadowIter = 32;             // The maximum number of shadow iterations permitted
uniform float marchDist = 50.0;          // The maximum distance to ray march
uniform float epsilon = 0.001;          // The error tolerance - lower means slower by better
//-------------------------------------------------

//--------------sdf uniforms----------------
uniform vec3 resolution;                 //screen resolution
uniform float time;                      //current time frame
uniform vec3 eyePos = vec3(0.0, 0.0, 1.0);      //eye position of the camera
uniform vec3 lookAt = vec3(0.0, 0.0, 0.0);      //where the camera is looking at

// Allows the user to set the shape type to be rendered. We have a sphere and an ellipsoid, extensions later, maybe.
uniform int shapeType = 2;

uniform float sphereR = 1.0;           // Parameter for radius of sphere
uniform vec3 ellipsoidR = vec3(0.1,0.1,0.05);      // Radius in 3D Vector for ellipsoid

const int shapeCount = 1;                // The number of shapes to use (needs to be const)
vec3 shapePos[shapeCount];               // Array of shape positions
mat3 shapeDir[shapeCount];               // Array of shape directions

//----------------------------------------------------------------------------


// All sdf primitives found in : http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdfSphere( vec3 p, float r )
{
  return length(p)-r;
}

float sdfEllipsoid( in vec3 p, in vec3 r )
{
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}
//-----------------------------------------------------------------------------------------------

// A basic function to return the signed distance from the ground plane
float ground(vec3 p)
{
    return p.y;
}

// define sdf shapes here
float shape(vec3 p)
{
    vec2 q;
    vec3 d;
    // switch is kept for future needs if need be.
    switch(shapeType) {
        case 1: // signed Ellipsoid
            return sdfEllipsoid(p, ellipsoidR);
        default: // A sphere
            return sdfSphere(p,sphereR);
    }

}
// Set shape animation
void setShape(float index, out vec3 shapePos, out mat3 shapeDir)
{
    float t = tau * mod(index * 0.2 + 0.02 * time + 0.12, 1.0);
    float a = 2.0*t;
    float b = 3.0*t;
    float c = 7.0 * t;
    shapePos = vec3(1.8 * cos(b),  1.0 + sin(a), 1.8 * cos(c));
    shapeDir = mat3(cos(a), -sin(a), 0.0, sin(a), cos(a), 0.0, 0.0, 0.0, 1.0);
    shapeDir *= mat3(cos(b), 0.0, -sin(b), 0.0, 1.0, 0.0, sin(b), 0.0, cos(b));
    shapeDir *= mat3(cos(c), -sin(c), 0.0, sin(c), cos(c), 0.0, 0.0, 0.0, 1.0);
}
/** This function initialises each shape in the scene. It is modified from https://www.shadertoy.com/view/XlXyD4
  * to support an arbitrary number of shapes
  */
void setScene()
{
    float shapeRatio = 5.0 / float(shapeCount);
    for (int i = 0; i < shapeCount; ++i)
    {
        setShape(float(i) * shapeRatio, shapePos[i], shapeDir[i]);
    }
}
float scene(vec3 p)
{
    float s = ground(p);
    for (int i = 0; i < shapeCount; ++i)
    {
        s = min(s, shape(shapeDir[i] * (p - shapePos[i])));
    }
    return s;
}

/** This is where the magic happens. The algorithm was taken from https://www.shadertoy.com/view/XlXyD4  but it is
  * quite generic and was originally taken from (and explained rather well) here:
  * http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
  */
float march(vec3 eye, vec3 dir)
{
    float depth = 0.0;
    for (int i = 0; i < marchIter; ++i)
    {
        float dist = scene(eye + depth * dir);
        depth += dist;
        if (dist < epsilon || depth >= marchDist) break;
    }
    return depth;
}

/** An approximation of a local surface normal at point p based on finite differences. Pretty generic, but this
  * version taken from here https://www.shadertoy.com/view/XlXyD4 .
  * Note this works anywhere in space, not just on the surface.
  */
vec3 normal(vec3 p) {
    return normalize(vec3(
        scene(vec3(p.x + epsilon, p.y, p.z)) - scene(vec3(p.x - epsilon, p.y, p.z)),
        scene(vec3(p.x, p.y + epsilon, p.z)) - scene(vec3(p.x, p.y - epsilon, p.z)),
        scene(vec3(p.x, p.y, p.z + epsilon)) - scene(vec3(p.x, p.y, p.z - epsilon))
    ));
}

/** A generic function to determine the ray to march into the scene based on the fragment coordinates.
  * Taken from https://www.shadertoy.com/view/XlXyD4
  */
vec3 ray(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size * 0.5;
    float z = fieldOfView * size.y;
    return normalize(vec3(xy, -z));
}


// Scene Lighting, Shadows and AOs-------------------------------------------------------------------

/** This is an implementation of ambient occlusion method for SDF scenes,
  * described here: http://iquilezles.org/www/material/nvscene2008/rwwtt.pdf .
  * Note this is considerably faster than the method implemented here: https://www.shadertoy.com/view/XlXyD4
  * but also demonstrates better quality for most shapes tested. Not sure why this other method is used.
  */


float ao(vec3 p, vec3 n) {
    float sum = 0.0;
    float factor = 1.0;
    float depthInc = aoDist / (aoIter+1);
    float depth = depthInc;
    float constK = 0.5; // Not sure what this needs to be, but 0.5 seems to work

    for (int i = 0; i < aoIter; ++i) {
        sum += factor * (depth - scene(p+n*depth)) / depth;
        factor *= 0.5;
        depth += depthInc;
    }
    return 1.0 -  constK * max(sum, 0.0);
}
// From http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
float hardShadow( in vec3 p, in vec3 dir, in float maxt) {
    int iter = 0;
    float t=epsilon;
    for(; (t < maxt) && (iter < shadowIter); ++iter)
    {
        float dist = scene(p + dir*t);
        if( dist < epsilon )
            return 0.0;
        t += dist;
    }
    return 1.0;
}
float softShadow( in vec3 p, in vec3 dir, float maxt) {
    float res = 1.0;
    float t = epsilon;
    float k = 8.0; // This determines the size of the penumbra (bigger is softer)
    int iter = 0;
    for( ; (t < maxt) && (iter < shadowIter); ++iter )
    {
        float h = scene(p + dir*t);
        if( h < epsilon )
            return 0.0;

        res = min( res, (k*h)/t );
        t += h;
    }
    return res;
}
/** A generic function to determine a view matrix.
  * Taken from https://www.shadertoy.com/view/XlXyD4
  */
mat3 viewMatrix(vec3 dir, vec3 up) {
    vec3 f = normalize(dir);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat3(s, u, -f);
}

vec3 Palette(in float t) {
    return vec3(0.5, 0.5, 0.5) + vec3(0.5, 0.5, 0.5)*cos( 6.28318*(vec3(1.0, 1.0, 1.0)*t+vec3(0.00, 0.10, 0.20)) );
}

//-------------------------------------------------------------------------------------------
void main ()
{

    // Determine where the viewer is looking based on the provided eye position and scene target
    vec3 dir = ray(2.5, resolution.xy, gl_FragCoord.xy);
    mat3 mat = viewMatrix(lookAt - eyePos, vec3(0.0, 1.0, 0.0));
    vec3 eye = eyePos;
    dir = mat * dir;

    setScene();

    // March until it hits the object. The depth indicates how far you have to travel down dir to get to the object.
    float depth = march(eye, dir);
    //skip forward if no geometries near march range
    if (depth >= marchDist - epsilon)
    {
        FragColour = vec4(depth);
        return;
    }

    // Calculate the intersection point and the associated normal
    vec3 p = eye + depth * dir;
    vec3 n = normal(p);


    // Shadow pass - just shoot a ray to the light
    vec3 lightPos = Light.Position.xyz; // eye
    vec3 pointToLight = lightPos - p;
    float distToLight = length(lightPos - p);
    vec3 pointToEye = eye - p;



    // Calculate the light and view vectors s and v respectively, along with the reflection vector
    vec3 s = normalize(pointToLight);
    vec3 v = normalize(pointToEye);
    vec3 r = reflect( -s, n );


    // Precompute the dot products
    vec3 h = normalize(s + v);
    float NdotL = max(dot(n, s), 0.0);
    float NdotV = max(dot(n, v), 0.0);
    float NdotH = max(dot(n, h), 0.0);
    float VdotH = max(dot(v, h), 0.0);

    float specular = 0.0;
    if (NdotL > 0.0)
    {
        float mSquared = roughnessValue * roughnessValue;

        // geometric attenuation
        float NH2 = 2.0 * NdotH;
        float invVdotH = 1.0 / VdotH;
        float g1 = (NH2 * NdotV) * invVdotH;
        float g2 = (NH2 * NdotL) * invVdotH;
        float geoAtt = min(1.0, min(g1, g2));

        // roughness (or: microfacet distribution function)
        // beckmann distribution function
        float r1 = 1.0 / ( 4.0 * mSquared * pow(NdotH, 4.0));
        float r2 = (NdotH * NdotH - 1.0) / (mSquared * NdotH * NdotH);
        float roughness = r1 * exp(r2);

        // fresnel
        // Schlick approximation
        float fresnel = pow(1.0 - VdotH, 5.0);
        fresnel *= (1.0 - F0);
        fresnel += F0;

        specular = max(0.0, (fresnel * geoAtt * roughness) / (NdotV * NdotL * 3.14));
    }

    // Determine if in shadow
    float shadowFactor = softShadow(p + n*epsilon, s, distToLight);
    // The result of this function will be a value between 0 and 1 describing how much darker this fragment should be
    float aoFactor = ao(p, n);

    FragColour = vec4(Palette(aoFactor), 1.0);

    // Calculate the normal (this is the expensive bit in Phong)
    //vec3 n = normalize( FragmentNormal );

    // Calculate the eye vector
    //vec3 v = normalize(vec3(-FragmentPosition));

    //vec3 lookup;

    /*if (isReflect) {
        lookup = reflect(v,n);
    } else {
        lookup = refract(v,n,refractiveIndex);
    }
    */
    // The mipmap level is determined by log_2(resolution), so if the texture was 4x4,
    // there would be 8 mipmap levels (128x128,64x64,32x32,16x16,8x8,4x4,2x2,1x1).
    // The LOD parameter can be anything inbetween 0.0 and 8.0 for the purposes of
    // trilinear interpolation.

    // This call actually finds out the current LOD
    //float lod = textureQueryLod(envMap, lookup).x;

    // Determine the gloss value from our input texture, and scale it by our LOD resolution
    //float gloss = (1.0 - texture(glossMap, FragmentTexCoord*2).r) * float(envMaxLOD);

    // This call determines the current LOD value for the texture map
    //vec4 colour = textureLod(envMap, lookup, gloss);

    // This call just retrieves whatever you want from the environment map
    //vec4 colour = texture(envMap, lookup);

    //FragColour = colour;
}

