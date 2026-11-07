#pragma once
#if defined(_MSC_VER) 
    #define MATH_VECTORCALL __vectorcall // Enables vectorcall calling convention, only in msvc
#else 
    #define MATH_VECTORCALL 
#endif