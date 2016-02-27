# nmfgpu
*Computation of the Non-negative Matrix Factorization (NMF) on CUDA capable Hardware*

## Installation

### Prerequisites
- CUDA Toolkit 7.0 or newer
- CMake 3.2.0 or newer
- Platform type x64 (CUDA libraries like cuBLAS are only available for x64 platforms)
- C++11 compatible compiler (supported compiler depends on CUDA Toolkit version)
  - *Windows*: Visual Studio 2013
  - *Linux*: g++ 4.8.x or newer
  - *Mac OS X*: clang x.x.x
- *Optional*: Doxygen
- CUDA capable device with compute capabilities 3.0 (Kepler) or greater

### Instructions

1. First use __git__ to clone the repository into a new directory:  
  ```
  git clone https://github.com/razorx89/nmfgpu.git
  ```  
  If you don't have git installed on your system, then you can download the repository as a zip compressed archive and extract it.
2. #### Using the GUI interface  
  Switch into the newly created directory and run __cmake-gui__:
  ```
  cd nmfgpu
  cmake-gui
  ```
  Create a new build folder for an out-of-source build and press *Configure*. Select the appropriate toolchain depending on your operating system (see *Prerequisites*). When the configuration has finished, then make sure to set *CMAKE_INSTALL_PREFIX* to the location where the compiled library should be installed to. If you want to build the documentation from source, then enable *NMFGPU_BUILD_DOCUMENTATION* (requires Doxygen). Press *Generate* to generate the required build files.

  #### Using the command-line interface
  If you are using a console environment, then you can configure __cmake__ using console commands:
  ```
  cd nmfgpu
  mkdir build
  cd build
  cmake .. -G "<generator>"
  ```
  All supported CMake generators are listed in CMake's help. Specify the generator according to your operating system (see *Prerequisites*). For example under Linux the *Unix Makefiles* generator would be used to compile with g++:
  ```
  cmake .. -G "Unix Makefiles"
  ```
  If the generator has an optional configuration for the platform, then you must explicitly define the x64 platform. Otherwise the *FindCUDA.cmake* script won't be able to resolve library paths for CUDA libraries. For Visual Studio 2013 the appropriate generator would be the following:
  ```
  cmake .. -G "Visual Studio 12 2013 Win64"
  ```
  Furthermore specify the installation directory, where the compiled library should be installed to:
  ```
  cmake . -DCMAKE_INSTALL_PREFIX="/usr/local/nmfgpu/"
  ```
3. After the build files have been generated, the library must be build. This step highly depends on the chosen generator and operating system. In the following two common workflows are described.

  #### Building for Windows platforms
  Open the generated solution file *nmfgpu.sln* and compile the *ALL_BUILD* project. Wait until the compilation has successfully finished and compile the *INSTALL* project, which will install the library to the configured location.

  If you are using [MSBuild](https://msdn.microsoft.com/en-us/library/wea2sca5%28v=vs.90%29.aspx) to compile your projects automaticly, then start for example the *VS2013 x64 Native Tools Command Prompt* and type:
  ```
  msbuild ALL_BUILD.vcxproj
  msbuild INSTALL.vcxproj
  ```

  #### Building for Unix platforms
  Once the *Unix Makefiles* generator has generated make files in the build directory, the build process is started by invoking __make__:
  ```
  make
  ```
  If the compilation was successful the library can be installed to the configured location:
  ```
  make install
  ```
4. For ease of usage you should define an environment variable called __NMFGPU_ROOT__ which points to the location of the installed library (the path provided to *CMAKE_INSTALL_PREFIX*).

## About
