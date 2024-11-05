cmake_minimum_required(VERSION 3.12)

EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE MACHINE)
project(evc VERSION 0.1
        DESCRIPTION "An efficient emulator for scattering"
        LANGUAGES C CXX
        )
set(CMAKE_VERBOSE_MAKEFILE on)

if(DEFINED ENV{GSL_DIR})
    include_directories( $ENV{GSL_DIR}/include )
endif()
if(DEFINED ENV{MYLOCAL})
    include_directories( $ENV{MYLOCAL}/include )
endif()

###################
# compiler flags
###################
# Adapt to icc vs gcc
MESSAGE("CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
if(${CMAKE_C_COMPILER} MATCHES ".*icc")
    set(OMPFLAGS "-qopenmp -DOPENMP=1")
    set(NOMPFLAGS "-Xcompiler -qopenmp -DOPENMP=1")
    set(FASTMATH "-fp-model fast")
    set(NFASTMATH "-Xcompiler -fp-model -Xcompiler fast")
    set(NCCBIN "-ccbin /opt/intel/bin/icc")
else()
    set(OMPFLAGS "-fopenmp -DOPENMP=1")
    set(NOMPFLAGS "-Xcompiler -fopenmp -DOPENMP=1")
    set(FASTMATH "-ffast-math")
    set(NFASTMATH "-use_fast_math")
endif()

# set(GPERFTOOLS "1")  # switch on/off Google perftools

if(GPERFTOOLS)
    # set(PERFLIBS "-lprofiler -ltcmalloc") tcmalloc appears to be slower for eos
    set(PERFLIBS "-lprofiler")
    set(PERFFLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
endif()

set(cxx_flags -x c++)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")
# For libtcmalloc from gperftools which contains an improved malloc
set(CMAKE_CXX_STANDARD 17)

if(${MACHINE} STREQUAL ppc64le)
    set(mflags "-mcpu=native")
else()
    set(mflags "-march=native")
endif()
message("eos machine:" ${MACHINE} ", mflags=" ${mflags})
set(CMAKE_CXX_FLAGS "-g -std=c++17 -Wall ${OMPFLAGS} ${FASTMATH} ${mflags} -Dregister= ${PERFFLAGS} ")

set(CMAKE_C_FLAGS "-g -std=c99 ${MPI_CXX_COMPILE_OPTIONS} ${OMPFLAGS} ${FASTMATH} ${PERFFLAGS} ")
#set(CMAKE_CXX_FLAGS_DEBUG "-O1")
#set(CMAKE_C_FLAGS_DEBUG "-O1")
# with sanitizer: requires at least -O1
#set(CMAKE_CXX_FLAGS_DEBUG "-O1 -fsanitize=address -fsanitize=leak -fsanitize=null -fsanitize=bounds-strict -fsanitize=alignment")
#set(CMAKE_C_FLAGS_DEBUG "-O1 -fsanitize=address -fsanitize=leak -fsanitize=null -fsanitize=bounds-strict -fsanitize=alignment")


###################
# includes/links
###################

include_directories( $ENV{MYLOCAL}/include /usr/local/include ./include ./src)

link_directories( $ENV{MYLOCAL}/lib /usr/local/lib )
if(DEFINED ENV{GSL_DIR})
    link_directories( $ENV{GSL_DIR}/lib)
endif()


###################
# Source files
###################

set( PROJECT_SOURCES
    main.cpp
    src/Emulator.cpp
    src/localGt+.cpp
    src/solveRSE.cpp
    src/Grid.cpp
    src/phaseShift.cpp
    )


###################
# executable(s)
###################

# computation of the equation of state
add_executable( ${PROJECT_NAME} main.cpp ${PROJECT_SOURCES} )


###################
# libraries
###################

set( PROJECT_LIBRARIES
        gsl armadillo ${PERFLIBS})

target_link_libraries( ${PROJECT_NAME} ${PROJECT_LIBRARIES} )


# library for using the emulator externally
add_library(evcemulator STATIC ${PROJECT_SOURCES})
set_target_properties(evcemulator PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(evcemulator PROPERTIES SOVERSION 1)

set(PUBLIC_HEADERS
        ${PROJECT_SOURCE_DIR}/src/localGt+.h
        ${PROJECT_SOURCE_DIR}/src/phaseShift.h
        )

set_target_properties(evcemulator PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")
#target_include_directories(evcemulator PRIVATE .)
#target_include_directories(evcemulator PRIVATE include)
#target_include_directories(evcemulator PRIVATE src)

install(TARGETS evcemulator
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include )