

#define MI 0.2           // stability factor to compute dt
#define ARGS 11          // tokens in executable command

#define _DUMP       // execution summary dump
//#undef  _DUMP     // execution summary dump
//#define _ABSOR_SQUARE    // use square absortion zone
#undef  _ABSOR_SQUARE  // don't use absortion zone
//#define _ABSOR_SPHERE    // use sphere absortion zone
#undef  _ABSOR_SPHERE  // don't use sphereabsortion zone
#define _RANDOM_BDRY     // use random boundary
//#undef _RANDOM_BDRY     // don'tuse random boundary

//#define SIGMA 20.0     // value of sigma (infinity) on formula 7 of Fletcher's paper
//#define SIGMA  6.0     // value of sigma on formula 7 of Fletcher's paper
//#define SIGMA  1.5     // value of sigma on formula 7 of Fletcher's paper
#define SIGMA  0.75      // value of sigma on formula 7 of Fletcher's paper
#define MAX_SIGMA 10.0   // above this value, SIGMA is considered infinite; as so, vsz=0


