#version 430

smooth in vec2 FragmentUV;

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


// An input texture for gloss - I'm going to repurpose this for the roughness maps
uniform sampler2D GlossTexture;

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
// store our calculated factors for future use
struct LightingFactors
{
    vec3 outF;
    vec3 inF;
};
// stores the product of spec, shadow and AO factors of inside and outside of the marble
LightingFactors factor = LightingFactors(
            vec3(0),
            vec3(0)
            );

// set important material values

uniform float F0 = 1.0; // fresnel reflectance at normal incidence
//uniform float k = 0.5;  // fraction of diffuse reflection (specular reflection = 1 - k)


// This determines and stores the shape positions and directions
const int shapeCount = 1;                // The number of shapes to use (needs to be const)
vec3 shapePos[shapeCount];               // Array of shape positions
mat3 shapeDir[shapeCount];               // Array of shape directions

// Allows the user to set the shape type to be rendered. Currently 6 are supported.
uniform int maxShapeType = 2;
uniform int shapeType = 0;

// The following fixed shape parameters are needed as we support lots of shapes.
// These could be input parameters if needed
uniform float radius = 0.9;
uniform vec3 elipsoidRadius = vec3(0.1,0.25,1.1);

//Constant material values in https://www.shadertoy.com/view/4lB3D1
const float DENSITY_MIN = 0.1;
const float DENSITY_MAX = 0.1;
const vec3 MATERIAL_COLOR = vec3(0.4,0.7,1);
const vec3 AIR_COLOR = vec3(0.5,0.8,1)*0.1;

const vec3 SURFACE_COLOR = vec3(0.8,1.,0.9);
const float ID_FLOOR = 1.0;
const float ID_GLASS_WALL = 2.000;
const float ID_INSIDE = 0.0;
const float ETA = 0.4;
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

float sdfSphere(vec3 p, float r)
{
    return length(p)-r;
}

float sdfElipsoid(vec3 p)
{
    return (length( p/elipsoidRadius ) - radius) * min(min(elipsoidRadius.x,elipsoidRadius.y),elipsoidRadius.z);
}


vec3 opTwist( vec3 p )
{
    float c = cos(1.6*p.y);
    float s = sin(1.6*p.y);
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


/** A function to return a shape to render in the scene at point p.
  */
vec3 shape(vec3 p, int type) {
    vec3 q;
    // type of shape to construct, 0 for outside, 1 inside
    switch(type)
    {
        case 0:
            //vec3 sphere = vec3(sdfSphere(p,radius),ID_GLASS_WALL,ETA);

            return vec3(sdfSphere(p,radius),ID_GLASS_WALL,ETA);

        case 1:
            q = opTwist(p);
            //twist elipsoid, scaled with
            float scale = 0.3;
            //vec3 elip = vec3(scale*sdfElipsoid(q/scale),ID_INSIDE,ETA);

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
    float c = 5.0;
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

    s = opUnion(s, shape(shapeDir[0] * (p - shapePos[0]),0));

    //s.x = abs(s.x);
    return s;
}


vec3 sceneIn(in vec3 p) {
    vec3 s = vec3(ground(p), ID_FLOOR,-1.0);

    s = opUnion(s, shape(shapeDir[0] * (p - shapePos[0]),1));
    //s.y = ID_INSIDE;
    //s.x = abs(s.x);
    return s;
}

/** An approximation of a local surface normal at point p based on finite differences. Pretty generic, but this
  * version taken from here https://www.shadertoy.com/view/XlXyD4 .
  * Note this works anywhere in space, not just on the surface.
  */
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

/** This is where the magic happens. The algorithm was taken from https://www.shadertoy.com/view/XlXyD4  but it is
  * quite generic and was originally taken from (and explained rather well) here:
  * http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
  */
float depthMarchOut(vec3 eye, vec3 dir) {
    float depth = 0.0;
    for (int i = 0; i < marchIter; ++i) {
        float dist = 0.5*sceneOut(eye + depth * dir).x;
        depth += dist;
        if (dist < epsilon || depth >= marchDist)
                        break;
    }
    return depth;
}

float depthMarchIn(vec3 eye, vec3 dir) {
    float depth = 0.0;
    for (int i = 0; i < marchIter; ++i) {
        float dist = 0.5*sceneIn(eye + depth * dir).x;
        depth += dist;
        if (dist < epsilon || depth >= marchDist)
                        break;
    }
    return depth;
}

// marching and querying distances
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
// for inside of the marble
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
    vec3 n = normalIn(p-rd*(epsilon-dist.x));
    CP cp;
    cp = CP(depth, n, dist.y, p);

    return cp;
}
//-------------------------------------------------------------------------------

// get color of refracted rays
vec3 refractCaustic(vec3 p, vec3 rd, vec3 ld, inout float eta) {

    vec3 cl = vec3(1);
    vec3 normal;
    //noisy shadow, ignore for now
    for(int j = 0; j < 2; ++j) {

        // call march function for refraction checks
        CP cp = findIntersectionIn(p, rd);
        if (length(cp.p) > 0) {
            break;
        }

        normal = sign(dot(rd, cp.normal))*cp.normal;
        rd = refract(rd, -normal, eta);
        eta = 1./eta;
        p = cp.p;
        p += normal*epsilon*2.;

    }
     //cl += normal*5 *mix(factor.inF,factor.outF,ETA);//mix(factor.outF,factor.inF,1-ETA)*(abs(dot(rd, cp.normal)));
     float d = clamp( dot( rd, ld ), 0.0, 1.0 );
     return smoothstep(0.9, 1.0, d)*cl;
}

vec3 caustic(vec3 p,vec3 ld, Ray ray) {
    vec3 VX = normalize(cross(ld, vec3(0,1,0)));
    vec3 VY = normalize(cross(ld, VX));
    vec3 c = vec3(0);
    vec3 cl;
    const int N =3;
    p += ray.cp.normal*epsilon;

    for(int i = 0; i < N;++i) {

        float n1 = rand(p.xz*10. + vec2(2. +float(i)*123.));
        float n2 = rand(p.xz*15. +vec2(3. +float(i)*111.));

        vec3 rd = ld+(VX*(n1-0.5)+VY*(n2-0.5))*0.1;
        //rd = ld;
        rd = normalize(rd);
        cl = refractCaustic(p, rd, ld, ray.eta);
        c += cl* dot(rd,ray.cp.normal );
    }
    return c*5/float(N);
}


// lightning and coloring is based on https://www.shadertoy.com/view/Xds3zN
vec3 getFloorColor(in Ray ray) {

    vec3 col = vec3(0);
    vec3 pos = ray.cp.p;
    vec3 ref = reflect( ray.rd, ray.cp.normal );
    // Checkerboard map
    float f = mod( floor(5.0*pos.z) + floor(5.0*pos.x), 2.0);
    col = 0.4 + 0.1*f*vec3(1.0);

    //col = col / (col + vec3(1.0));
    //col = pow(col, vec3(1.0/2.2));
    //vec3 LIGHT_DIR = normalize(vec3(-0.3,1.5,-0.1));
    vec3 LIGHT_DIR = normalize(Light.Position.xyz-pos);
    float dif = clamp( dot( ray.cp.normal, LIGHT_DIR ), 0.0, 1.0 );
    // if ray reaches the stuff inside the marble, do a different colour
    if(ray.cp.mat == ID_INSIDE)
    {
      return dif+vec3(0.02,0.02,0.8);
    }
    vec3 brdf = vec3(0);
    brdf += caustic(pos, LIGHT_DIR, ray);
    brdf += 1.2*dif*vec3(1.00,0.90,0.60);
    col = col*brdf;
    // exclude branching
    col *= (ID_GLASS_WALL-ray.cp.mat);

    return col;
}


vec3 getColor(in Ray ray) {

    vec3 c1 = ray.col * SURFACE_COLOR;

    vec3 c2 = getFloorColor(ray);
    // exclude branching
    return mix(c2, c1, ray.cp.mat-ID_FLOOR)+2.0*mix(factor.inF,factor.outF,(ray.eta - ETA)/(1./ETA-ETA));

}

vec3 getRayColor(Ray ray) {


    float d = mix(DENSITY_MIN, DENSITY_MAX, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 matColor = mix(AIR_COLOR, MATERIAL_COLOR, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 col = getColor(ray);

    float q = exp(-d*ray.cp.dist);
    col = col*q+matColor*(1.-q);
    return col*ray.share ;
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

// set of "recursion" functions, these simulate light bouncing inside the glass marble
// taken from https://www.shadertoy.com/view/4lB3D1

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

vec3 castRay(vec3 p, vec3 dir) {

    CP cp = findIntersection(p, dir);
    Ray ray = Ray( dir, cp, vec3(0), 1.0, ETA);

    rec3(ray);
    ray.col = getRayColor(ray);

        return ray.col;

}
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
vec3 softShadow( in vec3 p, in vec3 dir, float maxt) {
    vec3 res = vec3(1.0);
    float t = epsilon;
    float brightness = 1.3;
    float k =7.0; // This determines the size of the penumbra (bigger is softer)
    int iter = 0;
    float lastT;
    // call march function for refraction checks
    vec3 shadowCol = mix(vec3(0),castRay(p,dir),ETA);
    for( ; (t < maxt) && (iter < shadowIter); ++iter )
    {

        float h = sceneIn(p + dir*t).x;

        // solid shadow region
        if( h < epsilon )
        {
            // trace against for shadow caustics
            //return vec3(0);
            return mix(vec3(0),(brightness/t)*shadowCol,1-ETA);
        }

        res = min( res, vec3(((k*h)/t)) );
        t += h;
        lastT=t;
    }

    return (brightness-1)*normalize(res*shadowCol);
}
vec3 softShadowIn( in vec3 p, in vec3 dir, float maxt) {
    float res = 1.0;
    float t = epsilon;
    float k =7.0; // This determines the size of the penumbra (bigger is softer)

    int iter = 0;

    // call march function for refraction checks
    vec3 shadowCol = mix(vec3(0),castRay(p,dir),ETA);
    for( ; (t < maxt) && (iter < shadowIter); ++iter )
    {

        float h = sceneOut(p + dir*t).x;

        // solid shadow region
        if( h < epsilon )
        {
            // trace against for shadow caustics

            return 3.0*mix(vec3(0),t*shadowCol,1-ETA);
        }

        res = min( res, (k*h)/t );
        t += h;

    }
    return mix(vec3(res),shadowCol,1-ETA);
}
// from: Richard Southern's rendering example - SDF with microfacets and softshadow

// returns spec factor and NdotL
vec2 specular(vec3 p, vec3 n, vec3 dir,float depth)
{

  //float roughnessValue = 0.045; // 0 : smooth, 1: rough

  float roughnessValue = 0.1*texture(GlossTexture,FragmentUV).r; // 0 : smooth, 1: rough
  //get direction vector from point to light and to eye
  vec3 pointToLight = Light.Position.xyz - p;
  vec3 pointToEye = eyepos - p;

  // Calculate the light and view vectors s and v respectively, along with the reflection vector
  vec3 s = normalize(pointToLight);
  vec3 v = normalize(pointToEye);

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

//calls softshadow
vec3 shadowFactor(vec3 p, vec3 n, bool outOrIn)
{
    vec3 pointToLight = Light.Position.xyz - p;
    vec3 s = normalize(pointToLight);
    float distToLight = length(Light.Position.xyz - p);
    vec3 shadowFactor;
    //if you want shadow on the outside, have true
    if(outOrIn)
    {
        shadowFactor = softShadow(p + n*epsilon, s, distToLight);

        return shadowFactor;
    }
    shadowFactor = softShadowIn(p + n*epsilon, s, distToLight);

    return shadowFactor;
}



vec3 render(vec3 p, vec3 dir) {
    // getting values for specular
    float depthOut = depthMarchOut(eyepos,dir);
    float depthIn = depthMarchIn(eyepos,dir);
    vec3 pOut = p + depthOut * dir;
    vec3 nOut = normalOut(pOut);

    vec3 pIn = p + depthIn * dir;
    vec3 nIn = normalIn(pIn);

    //get specular with NdotL, shadow and ambient
    vec2 specOut = specular(pOut,nOut,dir,depthOut);
    vec2 specIn = specular(pIn,nIn,dir,depthIn);

    vec3 shadowOut = shadowFactor(pOut,nOut,true);
    vec3 shadowIn = shadowFactor(pIn,nIn,false);

    float aoOut = ao(pOut,nOut);
    float aoIn = ao(pIn,nIn);
    //float ratio = 1-ETA;
    //storing calculated factors to use later
    factor.outF = vec3(0.45*aoOut*shadowOut*(Light.Ld * specOut.y + Light.Ls * specOut.x));
    factor.inF = vec3(0.2*shadowIn*aoIn*(Light.Ld * specIn.y + Light.Ls *specIn.x));
    //vec3 overallFactor = mix(outFactor,inFactor,ratio);

    vec3 col = castRay(p, dir);

    return col;
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

    vec3 c  = render(eye,dir);
    fragColor = vec4(c,1.0);

}
