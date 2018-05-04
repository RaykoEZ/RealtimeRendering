#version 430

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
//uniform vec3      iMouse;
uniform vec3 eyepos = vec3(0.0, 0.0, 1.0);  // Eye position
uniform vec3 target = vec3(0.0, 0.0, 0.0);// Target
out vec4 fragColor;                      // Output colour value from this shader

// Ray marching parameters
uniform int marchIter = 64;             // The maximum number of iterations for the ray marching algorithm
uniform int shadowIter = 32;             // The maximum number of shadow iterations permitted
uniform float marchDist = 40.0;          // The maximum distance to ray march
uniform float epsilon = 0.001;          // The error tolerance - lower means slower by better

// Ambient occlusion parameters
uniform int aoIter = 8;                  // The number of iterations for ambient occlusion. Higher is better quality.
uniform float aoDist = 1.0;              // The size of the ambient occlusion kernel
uniform float aoPower = 16.0;             // The exponent for the ao kernel - a larger power increases the fall-off

// Colouring mode (1=black&white, 2=AO visualisation, 3=depth visualisation, 4=lambert shading visualisation, 5=shadow visualisation)
uniform int colourMode = 1;

// Modeling parameters
uniform float blendRadius = 0.5f;        // The radius for the polynomial min blend function - higher means more blend
const float tau = 6.283185;              // A modeling constant


// Structure for holding light parameters
struct LightInfo {
    vec4 Position; // Light position in eye coords.
    vec3 La; // Ambient light intensity
    vec3 Ld; // Diffuse light intensity
    vec3 Ls; // Specular light intensity
};

// We'll have a single light in the scene with some default values
uniform LightInfo Light = LightInfo(
            vec4(10.0, 10.0, 10.0, 1.0),   // position
            vec3(0.2, 0.2, 0.2),        // La
            vec3(1.0, 1.0, 1.0),        // Ld
            vec3(1.0, 1.0, 1.0)         // Ls
            );

// set important material values
uniform float roughnessValue = 0.1; // 0 : smooth, 1: rough
uniform float F0 = 0.5; // fresnel reflectance at normal incidence
uniform float k = 0.5;  // fraction of diffuse reflection (specular reflection = 1 - k)

// Cosine based palette from http://iquilezles.org/www/articles/palettes/palettes.htm
uniform vec3 ColorPalette[4] = vec3[](vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), vec3(0.00, 0.10, 0.20));
vec3 Palette(in float t) {
    return ColorPalette[0] + ColorPalette[1]*cos( 6.28318*(ColorPalette[2]*t+ColorPalette[3]) );
}

// This determines and stores the shape positions and directions
const int shapeCount = 1;                // The number of shapes to use (needs to be const)
vec3 shapePos[shapeCount];               // Array of shape positions
mat3 shapeDir[shapeCount];               // Array of shape directions

// Allows the user to set the shape type to be rendered. Currently 6 are supported.
uniform int maxShapeType = 2;
uniform int shapeType = 0;

// The following fixed shape parameters are needed as we support lots of shapes.
// These could be input parameters if needed
uniform float radius = 1.0;
uniform vec3 elipsoidRadius = vec3(0.1,0.45,2.0);

//Constant material values in https://www.shadertoy.com/view/4lB3D1
const float DENSITY_MIN = 0.1;
const float DENSITY_MAX = 0.1;
const vec3 MATERIAL_COLOR = vec3(0.5,0.8,1);
const vec3 AIR_COLOR = vec3(0.5,0.8,1)*0.1;

const vec3 SURFACE_COLOR = vec3(0.8,1.,0.9);
const float ID_FLOOR = 1.0;
const float ID_GLASS_WALL = 2.000;
const float ID_INSIDE = 0.500;
const float ETA = 0.45;
//  Data for raymarching and caustics. Sampled from https://www.shadertoy.com/view/4lB3D1
struct CP {
    float dist;
    vec3 normal;
    float mat;
    vec3 p;
};


struct Ray {
    vec3 rd;
    CP cp;
    vec3 col;
    float share;
    float eta;
};

float rand(vec2 n) {
        return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}


// A basic function to return the signed distance from the ground plane
float ground(vec3 p) {
    return p.y;
}


// All primitives and operations found here:
// http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

float sdfSphere(vec3 p)
{
    return length(p)-radius;
}

float sdfElipsoid(vec3 p)
{
    return (length( p/elipsoidRadius ) - radius) * min(min(elipsoidRadius.x,elipsoidRadius.y),elipsoidRadius.z);
}


vec3 opTwist( vec3 p )
{
    float c = cos(0.6*p.y);
    float s = sin(0.6*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y);
    return q;
}


vec3 opUnion( vec3 d1, vec3 d2 )
{

    return (d1.x<d2.x) ? d1 : d2;
}

vec3 opSubtract(  vec3 d1, vec3 d2 )
{
    return -d1.x>d2.x ? d2: d1;
}

/** An approximation of a local surface normal at point p based on finite differences. Pretty generic, but this
  * version taken from here https://www.shadertoy.com/view/XlXyD4 .
  * Note this works anywhere in space, not just on the surface.
  */

/** A function to return a shape to render in the scene at point p.
  */

vec3 shape(vec3 p, int type) {
    vec3 q;
    // type of shape to construct, 0 for outside, 1 inside
    switch(type)
    {
        case 0:
            return vec3(sdfSphere(p),ID_GLASS_WALL,ETA);

        case 1:
            q = opTwist(p);
            //twist elipsoid, scaled with
            float scale = 0.3;
            return vec3(scale*sdfElipsoid(q/scale),ID_INSIDE,ETA);
    }
}
// polynomial smooth min (k = 0.1) from http://iquilezles.org/www/articles/smin/smin.htm
// the bigger the k, the bigger the region of smoothing
float smin( float a, float b, float k ) {
    float h = clamp( 0.5*(1.0+(b-a)/k), 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

/** The rotation of the shapes was taken from here https://www.shadertoy.com/view/XlXyD4 and forms the basis of
  * the application.
  */
void setShape(float index, out vec3 shapePos, out mat3 shapeDir) {
    float t = tau * mod(index * 0.2 + 0.02 * iTime + 0.12, 1.0);
    float a = 0.0;
    float b = 3.0 ;
    float c = 7.0*t ;
    shapePos = vec3(1.8 * cos(b),  1.0 + sin(a), 1.8 * cos(c));
    shapeDir = mat3(cos(a), -sin(a), 0.0, sin(a), cos(a), 0.0, 0.0, 0.0, 1.0);
    shapeDir *= mat3(cos(b), 0.0, -sin(b), 0.0, 1.0, 0.0, sin(b), 0.0, cos(b));
    shapeDir *= mat3(cos(c), -sin(c), 0.0, sin(c), cos(c), 0.0, 0.0, 0.0, 1.0);
}

/** This function initialises each shape in the scene. It is modified from https://www.shadertoy.com/view/XlXyD4
  * to support an arbitrary number of shapes
  */
void setScene() {
    float shapeRatio = 5.0 / float(shapeCount);
    for (int i = 0; i < shapeCount; ++i) {

        setShape(float(i) * shapeRatio, shapePos[i], shapeDir[i]);
    }
}

/** This function returns the distance from a 3D point p to the surface. It is used a lot in
  * the ray marching algorithm, and needs to be as fast as humanly possible.
  */

//map >> map1

vec3 sceneOut(in vec3 p) {
    vec3 s = vec3(ground(p), ID_FLOOR,-1.0);
    /*
    for (int i = 0; i < shapeCount; ++i)
    {
        for (int j =0; j< maxShapeType; ++j)
        {
            s = opUnion(s, shape(shapeDir[i] * (p - shapePos[i]),j));

        }

    }*/
    s = opUnion(s, shape(shapeDir[0] * (p - shapePos[0]),0));

    //s.x = abs(s.x);
    return s;
}


vec3 sceneIn(in vec3 p) {
    vec3 s = vec3(ground(p), ID_FLOOR,-1.0);
    /*
    for (int i = 0; i < shapeCount; ++i)
    {
        for (int j =0; j< maxShapeType; ++j)
        {
            s = opUnion(s, shape(shapeDir[i] * (p - shapePos[i]),j));

        }

    }*/
    s = opUnion(s, shape(shapeDir[0] * (p - shapePos[0]),1));
    s.z = ID_INSIDE;
    //s.x = abs(s.x);
    return s;
}


vec3 normalOut(vec3 p) {
    return normalize(vec3(
        sceneOut(vec3(p.x + epsilon, p.y, p.z)).x - sceneOut(vec3(p.x - epsilon, p.y, p.z)).x,
        sceneOut(vec3(p.x, p.y + epsilon, p.z)).x - sceneOut(vec3(p.x, p.y - epsilon, p.z)).x,
        sceneOut(vec3(p.x, p.y, p.z + epsilon)).x - sceneOut(vec3(p.x, p.y, p.z - epsilon)).x
    ));
}
vec3 normalIn(vec3 p) {
    return normalize(vec3(
        sceneIn(vec3(p.x + epsilon, p.y, p.z)).x - sceneIn(vec3(p.x - epsilon, p.y, p.z)).x,
        sceneIn(vec3(p.x, p.y + epsilon, p.z)).x - sceneIn(vec3(p.x, p.y - epsilon, p.z)).x,
        sceneIn(vec3(p.x, p.y, p.z + epsilon)).x - sceneIn(vec3(p.x, p.y, p.z - epsilon)).x
    ));
}


/** I need another version of ths scene function to visualise the distance field on the ground plane.
  */
/*
float sceneWithoutGround(vec3 p) {
    float s = marchDist; // this needs to be bigger than the predicted distance to the scene
    for (int i = 0; i < shapeCount; ++i)
    {

        s = min(s, shape(shapeDir[i] * (p - shapePos[i])));
    }
    return s;
}
*/
/** This is where the magic happens. The algorithm was taken from https://www.shadertoy.com/view/XlXyD4  but it is
  * quite generic and was originally taken from (and explained rather well) here:
  * http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
  */
float depthMarch(vec3 eye, vec3 dir) {
    float depth = 0.0;
    for (int i = 0; i < marchIter; ++i) {
        float dist = sceneOut(eye + depth * dir).x;
        depth += dist;
        if (dist < epsilon || depth >= marchDist)
                        break;
    }
    return depth;
}



CP findIntersection(inout vec3 p, inout vec3 rd) {

    float depth = 0.0;

    vec3 dist;
    for (int i = 0; i < marchIter; ++i)
    {
        dist = sceneOut(p + depth * rd);
        depth += dist.x;
        if (dist.x < epsilon || depth >= marchDist) break;
    }


    p += rd * depth;
    // calculate normal in the father point to avoid artifacts
    vec3 n = normalOut(p-rd*(epsilon-dist.x));
    CP cp;
    cp = CP(depth, n, dist.y, p);

    return cp;
}
CP findIntersectionIn(inout vec3 p, inout vec3 rd) {

    float depth = 0.0;

    vec3 dist;
    for (int i = 0; i < marchIter; ++i)
    {
        dist = sceneIn(p + depth * rd);
        depth += dist.x;
        if (dist.x < epsilon || depth >= marchDist) break;
    }


    p += rd * depth;
    // calculate normal in the father point to avoid artifacts
    vec3 n = normalOut(p-rd*(epsilon-dist.x));
    CP cp;
    cp = CP(depth, n, dist.y, p);

    return cp;
}
//-------------------------------------------------------------------------------

float shadowFactor(vec3 p, vec3 n);
vec3 refractCaustic(vec3 p, vec3 rd, vec3 ld, inout float eta) {

    vec3 cl = vec3(1);
    for(int j = 0; j < 2; ++j) {

        CP cp = findIntersection(p, rd);
        if (length(cp.p) > 2.) {
            break;
        }


        vec3 normal = sign(dot(rd, cp.normal))*cp.normal;
        float shadow = shadowFactor(p,normal);
        cl *= SURFACE_COLOR;//*(abs(dot(rd, cp.normal)));
        rd = refract(rd, -normal, eta);
        eta = 1./eta;
        p = cp.p;
        p += normal*epsilon*2.;

    }
     float d = clamp( dot( rd, ld ), 0.0, 1.0 );
     return smoothstep(0.99, 1.0, d)*cl;
}

vec3 caustic(vec3 p,vec3 ld, Ray ray) {
    vec3 VX = normalize(cross(ld, vec3(0,1,0)));
    vec3 VY = normalize(cross(ld, VX));
    vec3 c = vec3(0);

    const int N =3;
    p += ray.cp.normal*epsilon;

    for(int i = 0; i < N;++i) {

        float n1 = rand(p.xz*10. + vec2(2. +float(i)*123.));
        float n2 = rand(p.xz*15. +vec2(3. +float(i)*111.));

        vec3 rd = ld+(VX*(n1-0.5)+VY*(n2-0.5))*0.1;
        //rd = ld;
        rd = normalize(rd);

        vec3 cl = refractCaustic(p, rd, ld, ray.eta);

        c += cl* dot(rd,ray.cp.normal);
    }
    return c*5./float(N);
}


// lightning is based on https://www.shadertoy.com/view/Xds3zN
vec3 getFloorColor(in Ray ray) {

    vec3 col = vec3(0);
    vec3 pos = ray.cp.p;
    vec3 ref = reflect( ray.rd, ray.cp.normal );
    // Checkerboard map
    float f = mod( floor(5.0*pos.z) + floor(5.0*pos.x), 2.0);
    col = 0.4 + 0.1*f*vec3(1.0);
    vec3 LIGHT_DIR = normalize(vec3(-0.3,1.5,-0.1));
    float dif = clamp( dot( ray.cp.normal, LIGHT_DIR ), 0.0, 1.0 );
    // if ray reaches the stuff inside the marble, do a different colour
    if(ray.cp.mat == ID_INSIDE)
    {
      return 1.20*dif+vec3(0.05,0.1,1.0);
    }
    vec3 brdf = vec3(0.0);
    brdf += caustic(pos, LIGHT_DIR, ray);
    brdf += 1.20*dif*vec3(1.00,0.90,0.60);
    col = col*brdf;
    // exclude branching
    col *= (ID_GLASS_WALL-ray.cp.mat);

    return col;
}


vec3 getColor(in Ray ray) {

    vec3 p = ray.cp.p ;// can be used by SURFACE_COLOR define
    vec3 c1 = ray.col * SURFACE_COLOR;
    vec3 c2 = getFloorColor(ray);
    // exclude branching
    return mix(c2, c1, ray.cp.mat-ID_FLOOR);

}

vec3 getRayColor(Ray ray) {


    float d = mix(DENSITY_MIN, DENSITY_MAX, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 matColor = mix(AIR_COLOR, MATERIAL_COLOR, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 col = getColor(ray);

    float q = exp(-d*ray.cp.dist);
    col = col*q+matColor*(1.-q);
    return col*ray.share;
}

void getRays(inout Ray ray, out Ray r1, out Ray r2) {
     vec3 p = ray.cp.p;
    float cs = dot(ray.cp.normal, ray.rd);
    // simple approximation
    float fresnel = 1.0-abs(cs);
    //fresnel = mix(0.1, 1., 1.0-abs(cs));
    float r = ray.cp.mat - ID_FLOOR;
    vec3 normal = sign(cs)*ray.cp.normal;
    vec3 refr = refract(ray.rd, -normal, ray.eta);
    vec3 refl = reflect(ray.rd, ray.cp.normal);
    vec3 z = normal*epsilon*2.;
    p += z;
    r1 = Ray(refr, findIntersectionIn(p, refr),  vec3(0.2,0.3,0.9),(1.-fresnel)*r, 1./ray.eta);
    p -= 2.*z;
    r2 = Ray( refl, findIntersection(p, refl), vec3(0),r*fresnel, ray.eta);
}

// set of "recursion" functions

void rec1(inout Ray ray) {
    ray.col += getRayColor(ray);
}


void rec2(inout Ray ray) {

    Ray r1,r2;
    getRays(ray, r1, r2);

    ray.col += getRayColor(r1);
    ray.col += getRayColor(r2);
}

void rec3(inout Ray ray) {

    Ray r1,r2;
    getRays(ray, r1, r2);

    rec2(r1);
    ray.col += getRayColor(r1);
    // use first level of relfection rays only to improve performance
    rec1(r2);
    ray.col += getRayColor(r2);
}


vec3 castRay(vec3 p, vec3 rd) {
    CP cp = findIntersection(p, rd);
    Ray ray = Ray( rd, cp, vec3(0), 1.0, ETA);
    rec3(ray);
    ray.col = getRayColor(ray);

        return ray.col;

}



vec3 render(vec3 p, vec3 rd) {
    vec3 col= castRay(p, rd);

    return col;
}




/** A generic function to determine the ray to march into the scene based on the fragment coordinates. 
  * Taken from https://www.shadertoy.com/view/XlXyD4 
  */
vec3 ray(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size * 0.5;
    float z = fieldOfView * size.y;
    return normalize(vec3(xy, -z));
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

/** A generic function to determine the rotation matrix necessary to align the view direction with the default ray.
  * Taken from https://www.shadertoy.com/view/XlXyD4 
  
mat3 alignMatrix(vec3 dir) {
    vec3 f = normalize(dir);
    vec3 s = normalize(cross(f, vec3(0.48, 0.6, 0.64)));
    vec3 u = cross(s, f);
    return mat3(u, s, f);
}*/

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
        sum += factor * (depth - sceneOut(p+n*depth).x) / depth;
        factor *= 0.5;
        depth += depthInc;
    }
    return 1.0 -  constK * max(sum, 0.0);
}

// From http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
float softShadow( in vec3 p, in vec3 dir, float maxt) {
    float res = 1.0;
    float t = epsilon;
    float k = 8.0; // This determines the size of the penumbra (bigger is softer)
    int iter = 0;
    for( ; (t < maxt) && (iter < shadowIter); ++iter )
    {
        float h = sceneOut(p + dir*t).x;

        if( h < epsilon )
            return 0.0;

        res = min( res, (k*h)/t );
        t += h;
    }
    return res;
}

// from: Richard Southern's rendering example - SDF with microfacets and softshadow

// returns spec factor and NdotL
vec2 specular(vec3 p, vec3 n, vec3 dir,float depth)
{

  //get direction vector from point to light and to eye
  vec3 pointToLight = Light.Position.xyz - p;
  vec3 pointToEye = eyepos - p;

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

  if (NdotL > 0.0) {
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

  return vec2(specular,NdotL);
}

float shadowFactor(vec3 p, vec3 n)
{
  vec3 pointToLight = Light.Position.xyz - p;
  vec3 s = normalize(pointToLight);
  float distToLight = length(Light.Position.xyz - p);
  float shadowFactor = softShadow(p + n*epsilon, s, distToLight);
  return shadowFactor;
}


void main() {

    // Determine where the viewer is looking based on the provided eye position and scene target
    vec3 dir = ray(2.0, iResolution.xy, gl_FragCoord.xy);
    vec2 uv = gl_FragCoord.xy / iResolution.xy-0.5;
    uv.x*=iResolution.x/iResolution.y;
    mat3 mat = viewMatrix(target - eyepos, vec3(0.0, 1.0, 0.0));
    vec3 eye = eyepos;
    dir = mat * dir;
    setScene();

    // getting values for specular
    float depth = depthMarch(eyepos,dir);
    vec3 p = eye + depth * dir;
    vec3 n = normalOut(p);

    //get specular with NdotL, shadow and ambient
    vec2 spec = specular(p,n,dir,depth);
    float shadow = shadowFactor(p,n);
    float ao = ao(p,n);
    //fragColor = vec4( c , 1. );
    vec3 c = render(eye,dir);

    fragColor = vec4(0.5*ao*shadow*(Light.Ld * spec.y + Light.Ls * spec.x)+c,1.0);

    /*
    // Initialise the scene based on the current elapsed time
    setScene();
    
    // March until it hits the object. The depth indicates how far you have to travel down dir to get to the object.
    float depth = depthMarch(eye, dir);
    if (depth >= marchDist - epsilon) {
        fragColor = vec4(depth);
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
    if (NdotL > 0.0) {
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

    switch(colourMode) {        
    case 2:
        fragColor = vec4(Palette(aoFactor), 1.0);
        break;
    case 3:
        //fragColor = vec4(Palette(sceneWithoutGround(p)),1.0);
        break;
    case 4:
        fragColor = vec4(Palette(NdotL), 1.0);
        break;
    case 5:
        fragColor = vec4(Palette(shadowFactor), 1.0);
        break;    
    default:    
        fragColor = vec4(aoFactor * shadowFactor * (Light.Ld * NdotL + Light.Ls * specular), 1.0);
        break;
    }
    */
}
